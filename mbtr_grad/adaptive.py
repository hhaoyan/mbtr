# adaptive grid search, originally written by Matthias Rupp from qmmlpack.
import math

import numpy as np

__all__ = [
    "QMMLException",
    "AdaptiveGridSearch",
]


class QMMLException(Exception):
    """Exception base class.

    All exceptions in qmmlpack derive from this class."""

    pass


def is_sequence(arg):
    """True if arg is a list, tuple, array or similar object, but not a string, dictionary or set."""
    # checking for attributes '__getitem__' or '__iter__' erroneously accepts NumPy floating point variables
    if isinstance(arg, dict) or isinstance(arg, set):
        return False
    try:
        iter(arg)
        return not isinstance(arg, str)
    except TypeError:
        return False


class AdaptiveGridSearch:
    """Optimization of several variables via local gradient descent on a logarithmic grid."""

    def _parse_variable(self, v):
        """Parses one variable.

        0  initial value
        1  priority
        2  step size
        3  min
        4  max
        5  direction
        6  base"""

        defaults = (0.0, 1, 1.0, -np.inf, +np.inf, 0, 2.0)
        keywords = (
            (0, "value"),
            (1, "priority"),
            (2, "stepsize"),
            (3, "min"),
            (3, "minimum"),
            (4, "max"),
            (4, "maximum"),
            (5, "direction"),
            (6, "base"),
        )

        # dictionary specification
        if isinstance(v, dict):
            vv = list(defaults)
            for (i, kw) in keywords:
                if kw in v:
                    vv[i] = v[kw]
            v = vv

        if not is_sequence(v):
            raise QMMLException("Invalid variable specification")

        # initial value
        iv = defaults[0] if len(v) < 1 else float(v[0])

        # priority
        p = defaults[1] if len(v) < 2 else int(v[1])
        if len(v) >= 2 and p != v[1]:
            raise QMMLException("Non-integer priority specified")
        if p <= 0:
            raise QMMLException("Invalid priority specified")

        # step size
        s = defaults[2] if len(v) < 3 else float(v[2])
        if s <= 0:
            raise QMMLException("Invalid step size specified")

        # minimum
        rmin = defaults[3] if len(v) < 4 else v[3]

        # maximum
        rmax = defaults[4] if len(v) < 5 else v[4]

        # direction
        d = defaults[5] if len(v) < 6 else int(v[5])
        if len(v) >= 6 and d != v[5]:
            raise QMMLException("Non-integer direction specified")
        if d not in (-1, 0, +1):
            raise QMMLException("Invalid direction specified")

        # base
        b = defaults[6] if len(v) < 7 else float(v[6])
        if b <= 0:
            raise QMMLException("Invalid base specified")

        # additional tests
        if rmin >= rmax:
            raise QMMLException("Invalid range specified")
        if not (rmin <= iv <= rmax):
            raise QMMLException("Initial value not in range")

        return [iv, p, s, rmin, rmax, d, b]

    def __init__(self, f, variables, resolution=None):
        """AdaptiveGridSearch(f, {v1,v2,...}, resolution=None) finds local minimum of f(v1,v2,...) via local gradient descent on a logarithmic grid.

        Parameters:
          f - function to be minimized
          v1,v2,... - variables, see below
          resolution - if specified as a number a, f(...) will be rounded to multiples of a

        A variable is a tuple {val, pri, stp, min, max, dir, b}, where
          val = initial value; real number; exponent to base b
          pri = priority; positive integer; higher priorities are optimized before lower ones; several variables may have the same priority
          stp = step size; positive real number; refers to exponent
          min = lowest allowed value for variable, val in [min,max]
          max = highest allowed value for variable, val in [min,max]
          dir = direction; either -1 (small values preferred), 0 (indifferent), +1 (large values preferred)
          b   = base; positive real number
          Later entries can be omitted. Default values are (0., 1, 1., (-np.inf,+np.inf), 0, 2., (True,True)).
          The logarithmized grid is the intersection between ...,i-2s,i-s,i,i+s,i+2s,... and (min,max).
          Alternatively, a dictionary with possible keys 'value', 'priority', 'stepsize', 'min', 'max', 'direction', 'base' can be specified.

        Once the grid search is initialized, optimization steps can be taken via step().
        Data member stats yields statistics on the optimization so far.

        If optimizing an expensive function, such as hyperparameters of a machine learning model, it is important that f uses cached values.
        For example, if optimizing performance of a kernel learning model, kernel matrices for different values of kernel hyperparameters should be cached where possible."""

        # the descent can not cross itself, but a visited list helps avoid retrying unsuccessful grid points, reducing number of steps.

        # initialize state
        self._vars = np.asarray([self._parse_variable(v) for v in variables])
        self._f = f
        self._resolution = None if resolution is None else float(resolution)

        self.num_vars = len(self._vars)
        self.num_steps = 0
        self.num_evals = 0
        self.update = False
        self.done = False
        self.conv = [False] * len(
            self._vars
        )  # indicates which variables to optimize; True entries are currently converged
        self.dirs = [
            np.random.choice([-1, 1]) if v[5] == 0 else int(v[5]) for v in self._vars
        ]
        self.cur_ind = 0  # index of variable currently being iterated
        self.cur_v = np.copy(self._vars[:, 0])  # current values of variables
        self.trial_v = np.empty_like(self.cur_v)  # last tried values of variables
        self.trial_f = None  # f of last tried values of variables
        self.best_v = np.copy(self.cur_v)  # best solution so far
        self.best_f = np.inf
        # f of best solution so far, f(best_v)
        self.visited = set()  # visited grid points so far

        # initialize optimization
        self._trial(0.0)
        self.update = False  # force determination of cur_ind

    def _trial(self, step):
        ind = self.cur_ind  # shorthand
        self.update = False

        # take step
        self.trial_v[:] = self.cur_v
        self.trial_v[ind] += step

        # check for range (clamping incurs some unnecessary calls of f, but allows variable to change direction)
        self.trial_v[ind] = max(
            self._vars[ind, 3], min(self.trial_v[ind], self._vars[ind, 4])
        )  # clamp to [min, max]

        # check if visited already
        if tuple(self.trial_v) in self.visited:
            return

        # call f
        trialpowers = np.power(self._vars[:, 6], self.trial_v)
        self.trial_f = self._f(*trialpowers)
        self.num_evals += 1
        self.visited.add(tuple(self.trial_v))
        if self._resolution is not None:
            # rounds to precision of resolution, then to number of significant digits to avoid trailing noise in last digits
            # https://stackoverflow.com/a/28427814/3248771
            self.trial_f = round(
                round(self.trial_f / self._resolution) * self._resolution,
                -int(math.floor(math.log10(self._resolution))),
            )

        # update if improvement
        if self.trial_f < self.best_f:
            self.cur_v[:] = self.trial_v
            self.best_v[:] = self.trial_v
            self.best_f = self.trial_f
            self.update = True
            self.conv = [False] * self.num_vars

    def step(self):
        """Takes one optimization step.

        The step taken can involve multiple evaluations of f."""

        if self.done:
            return

        # determine variable to optimize next
        if not self.update:
            self.cur_ind = [
                i for (i, d) in enumerate(self.conv) if not d
            ]  # variables currently being optimized
            self.cur_ind = [
                i
                for i in self.cur_ind
                if self._vars[i, 1] == max(self._vars[self.cur_ind, 1])
            ]  # variables with highest priority
            self.cur_ind = np.random.choice(self.cur_ind)

        # take one step into direction
        if self._vars[self.cur_ind, 5] in (-1, +1):
            self.dirs[self.cur_ind] = int(
                self._vars[self.cur_ind, 5]
            )  # if not, keep last direction
        self._trial(self.dirs[self.cur_ind] * self._vars[self.cur_ind, 2])

        # change direction if unsuccessful
        if not self.update:
            self.dirs[self.cur_ind] *= -1
            self._trial(self.dirs[self.cur_ind] * self._vars[self.cur_ind, 2])

        # if no update, variable is converged
        if not self.update:
            self.conv[self.cur_ind] = True

        # update statistics
        self.num_steps += 1

        # test termination condition
        if all(self.conv):
            self.done = True

    def __bool__(self):
        """Indicates whether optimization is ongoing or done."""
        return not self.done

    def __str__(self, verbose=False):
        """Returns formatted statistics."""
        s = self.stats()
        if verbose == "compact":
            descr = "#{}({}): {} -> {}  ({})".format(
                s["num_steps"],
                s["num_evals"],
                tuple(s["best_vals"]),
                s["best_f"],
                "".join(["T " if c else "F " for c in s["converged"]]),
            )
        else:
            descr = (
                "{}\n\n".format(self.__class__.__name__)
                + "#vars = {}, #steps = {}, #evals = {}, done = {}\n".format(
                    s["num_vars"], s["num_steps"], s["num_evals"], s["done"]
                )
                + "best solution: f{} = {}\n".format(tuple(s["best_vals"]), s["best_f"])
                + "var = {}, update = {}\n".format(s["cur_var"], s["update"])
                + "converged vars: {}, ".format(
                    " ".join(["T " if c else "F " for c in s["converged"]])
                )
                + "directions: {}\n".format(
                    " ".join(["{:+d}".format(d) for d in s["directions"]])
                )
                + (
                    ""
                    if not verbose
                    else "\nvisited:\n{}".format(
                        "".join(sorted(["  {}\n".format(v) for v in s["visited"]]))
                    )
                )
            )
        return descr

    def stats(self):
        """Returns dictionary with statistics on the course of the optimization so far.
        The following information will be contained in the list:

        * `variables` - variables settings
        * `resolution` - resolution
        * `num_vars` - number of variables (arity of f)
        * `num_steps` - number of times step() was called
        * `num_evals` - number of times f was evaluated
        * `update` - whether best solution was updated last step
        * `done` - True if optimization is finished
        * `converged` - which variables are currently in local minima
        * `directions` - current directions of variables
        * `cur_var` - index of currently optimized variable
        * `cur_vals` - current values of variables
        * `trial_vals` - last tried values of variables
        * `trial_f` - function value for trial_vals
        * `best_vals` - best found variable values so far
        * `best_f` - function value for best_vals
        * `visited` - set of variable values visited so far

        :returns: Dictionary contains the above information.
        """

        return {
            "variables": self._vars,
            "resolution": self._resolution,
            "num_vars": self.num_vars,
            "num_steps": self.num_steps,
            "num_evals": self.num_evals,
            "update": self.update,
            "done": self.done,
            "converged": self.conv,
            "directions": self.dirs,
            "cur_var": self.cur_ind,
            "cur_vals": self.cur_v,
            "trial_vals": self.trial_v,
            "trial_f": self.trial_f,
            "best_vals": self.best_v,
            "best_f": self.best_f,
            "visited": self.visited,
        }
