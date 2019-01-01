#include <Python.h>
#include "presets.h"

/*
 * Compute MBTR from molecular/crystal structures (ver 0.01).
 * Publication:
 *   Haoyan Huo and Matthias Rupp, Unified Representation for Machine Learning of
 *   Molecules and Crystals, arXiv:1704.06439 (2017).
 *   Please also refer to http://qmml.org/
 * ===============================================================================
 *   Usage: %s -molecular -dataset <dataset> -output <output> [args]
 *   Usage: %s -periodic -dataset <dataset> -output <output> [args]
 *
 *   Args:
 *       <dataset>    - file name of dataset (xyz format).
 *       <output>     - file name of MBTR output file (numpy array I/O format).
 *       -sparse      - generate sparse matrix using scipy CSR Sparse definition.
 *       -preset id   - use one of the following presets, see below.
 *       -start float - the beginning of the interval for the last dimension of MBTR.
 *       -end float   - the ending of the interval for the last dimension of MBTR.
 *                      'start' and 'end' will be automatically determined if not present (but slower).
 *       -rank uint   - tensor rank (default:2).
 *       -sigma float - smearing width (default:0.02).
 *       -grid uint   - grid size for the last dimension of MBTR (default:20).
 *       -D float     - parameter D in weighting funciton (default:4.0).
 *   Note:
 *       1. In either case, atom coordinates must be cartesian.
 *       2. For periodic xyz file, one line containing basis vectors (nine numbers)
 *          should be inserted before atom lines.
 *   Presets:
 * */

std::vector<uint> AtomNumberList(std::vector<System> &systems) {
    std::vector<uint> an;
    for (const auto &system:systems) {
        for (auto i:system.atoms) {
            if (an.end() == std::find(an.begin(), an.end(), i.atom_number)) {
                an.push_back(i.atom_number);
            }
        }
    }
    std::sort(an.begin(), an.end());
    return an;
}

#define RaiseKeyErrorAndReturnFalse(x) {PyErr_SetString(PyExc_KeyError, x); return false;}
#define RaiseTypeErrorAndReturnFalse(x) {PyErr_SetString(PyExc_TypeError, x); return false;}
#define RaiseValueErrorAndReturnFalse(x) {PyErr_SetString(PyExc_ValueError, x); return false;}
#define GetDictItemWithError(dict, item, target, error) {\
    if((target = PyDict_GetItemString(dict, item)) == nullptr)  \
    RaiseKeyErrorAndReturnFalse(error); \
}
#define CheckTypeWithError(item, type_checker, error) {\
    if(!type_checker(item))RaiseTypeErrorAndReturnFalse(error); \
}

bool ReadConfig(PyObject *config, PresetInfo &preset) {
    CheckTypeWithError(config, PyDict_Check, "config must be a dict");

    PyObject *value;

    GetDictItemWithError(config, "preset", value, "Config dict must have 'preset'");
    CheckTypeWithError(value, PyLong_Check, "'preset' must be an integer");
    preset.preset = PyLong_AsLong(value);

    GetDictItemWithError(config, "rank", value, "Config dict must have 'rank'");
    CheckTypeWithError(value, PyLong_Check, "'rank' must be an integer");
    preset.rank = PyLong_AsUnsignedLong(value);

    GetDictItemWithError(config, "grid_size", value, "Config dict must have 'grid_size'");
    CheckTypeWithError(value, PyLong_Check, "'grid_size' must be an integer");
    preset.grid_size = PyLong_AsUnsignedLong(value);

    GetDictItemWithError(config, "is_periodic", value, "Config dict must have 'is_periodic'");
    CheckTypeWithError(value, PyBool_Check, "'is_periodic' must be a boolean");
    preset.is_periodic = (value == Py_True);

    GetDictItemWithError(config, "sigma", value, "Config dict must have 'sigma'");
    CheckTypeWithError(value, PyFloat_Check, "'sigma' must be a float");
    preset.sigma = PyFloat_AsDouble(value);

    GetDictItemWithError(config, "D", value, "Config dict must have 'D'");
    CheckTypeWithError(value, PyFloat_Check, "'D' must be a float");
    preset.D = PyFloat_AsDouble(value);

    GetDictItemWithError(config, "is_range_set", value, "Config dict must have 'is_range_set'");
    CheckTypeWithError(value, PyBool_Check, "'is_range_set' must be a boolean");
    preset.is_range_set = (value == Py_True);

    if (preset.is_range_set) {
        GetDictItemWithError(config, "tensor_range_min", value, "Config dict must have 'tensor_range_min'");
        CheckTypeWithError(value, PyFloat_Check, "'tensor_range_min' must be a float");
        preset.tensor_range.first = PyFloat_AsDouble(value);

        GetDictItemWithError(config, "tensor_range_max", value, "Config dict must have 'tensor_range_max'");
        CheckTypeWithError(value, PyFloat_Check, "'tensor_range_max' must be a float");
        preset.tensor_range.second = PyFloat_AsDouble(value);
    }

    return true;
}

bool UnpackSystem(PyObject *data, System &system) {
    CheckTypeWithError(data, PyDict_Check, "'system' must be a dict");

    PyObject *value;

    // Fill periodic flag
    GetDictItemWithError(data, "is_periodic", value, "'system' must have 'is_periodic' key");
    CheckTypeWithError(value, PyBool_Check, "'is_periodic' must be a boolean");
    system.is_periodic = (value == Py_True);

    // Fill the basis vectors
    if (system.is_periodic) {
        GetDictItemWithError(data, "basis_vector", value, "'system' must have 'basis_vector' key");
        CheckTypeWithError(value, PyTuple_Check, "'basis_vector' must be a tuple");
        if (PyTuple_Size(value) != 9) {
            PyErr_SetString(PyExc_ValueError, "'basis_vector' must be a 9-sized tuple");
            return false;
        }
        double basis_values[9];
        for (int i = 0; i < 9; i++) {
            PyObject *_value = PyTuple_GetItem(value, i);
            CheckTypeWithError(_value, PyFloat_Check, "'basis_vector' must only contain floats");
            basis_values[i] = PyFloat_AsDouble(_value);
        }
        system.SetBasis(basis_values);
    }

    // Fill atoms
    GetDictItemWithError(data, "atoms", value, "'system' must have 'atoms' key");
    CheckTypeWithError(value, PyList_Check, "'atoms' must be a list");
    auto n_atoms = PyList_Size(value);
    if (n_atoms <= 0) {
        PyErr_SetString(PyExc_ValueError, "'atoms' must not be empty list");
        return false;
    }
    for (auto i = 0; i < n_atoms; i++) {
        PyObject *atom = PyList_GetItem(value, i);
        CheckTypeWithError(atom, PyTuple_Check, "'atoms' must be list of tuples");
        if (PyTuple_Size(atom) != 4) {
            PyErr_SetString(PyExc_ValueError, "Each atom must be a 4-sized tuple");
            return false;
        }

        PyObject *items[4] = {
                PyTuple_GetItem(atom, 0),
                PyTuple_GetItem(atom, 1),
                PyTuple_GetItem(atom, 2),
                PyTuple_GetItem(atom, 3)
        };
        CheckTypeWithError(items[0], PyLong_Check, "first element of atom tuple must be an integer");
        CheckTypeWithError(items[1], PyFloat_Check, "second element of atom tuple must be a float");
        CheckTypeWithError(items[2], PyFloat_Check, "third element of atom tuple must be a float");
        CheckTypeWithError(items[3], PyFloat_Check, "fourth element of atom tuple must be a float");

        uint atom_number = static_cast<uint>(PyLong_AsUnsignedLong(items[0]));
        double position[3] = {
                PyFloat_AsDouble(items[1]),
                PyFloat_AsDouble(items[2]),
                PyFloat_AsDouble(items[3])
        };

        system.atoms.emplace_back(atom_number, position[0], position[1], position[2]);
    }

    return true;
}

bool UnpackSystems(PyObject *data, std::vector<System> &systems) {
    CheckTypeWithError(data, PyList_Check, "'systems' must be a list");

    auto n_atoms = PyList_Size(data);
    if (n_atoms <= 0) {
        RaiseValueErrorAndReturnFalse("'systems' must not be an empty list");
    }

    for (auto i = 0; i < n_atoms; i++) {
        System system;
        if (!UnpackSystem(PyList_GetItem(data, i), system))
            return false;
        systems.push_back(system);
    }

    return true;
}

bool ParseArgs(PyObject *args, PresetInfo &preset_info) {
    PyObject *systems_data, *config;

    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyList_Type, &systems_data,
                          &PyDict_Type, &config)) {
        return false;
    }

    if (!ReadConfig(config, preset_info) || !UnpackSystems(systems_data, preset_info.systems)) {
        return false;
    }

    // Sanity check
    preset_info.z = AtomNumberList(preset_info.systems);
    for (const auto &system: preset_info.systems) {
        if (system.is_periodic != preset_info.is_periodic) {
            PyErr_SetString(PyExc_ValueError, "Each system must have save 'is_periodic' flag as in the config");
            return false;
        }
    }

    return true;
}


static PyObject *compute(PyObject *, PyObject *args) {
    PresetInfo info;
    if (!ParseArgs(args, info)) {
        return nullptr;
    }

    if (!info.is_range_set) {
        PyErr_SetString(PyExc_ValueError, "'is_range_set' must be True to calculate MBTR");
        return nullptr;
    }

    std::vector<MBTRResult> results;
    if (!PresetsComputeMBTR(info, results)) {
        return nullptr;
    }

    PyObject *result_list = PyList_New(0);
    if (result_list == nullptr)
        return nullptr;

    for (auto &result : results) {
        std::vector<uint> &atom_numbers = result.atom_number;
        PyObject *atom_numbers_list = PyList_New(0);

        if (atom_numbers_list == nullptr) {
            Py_DecRef(result_list);
            return nullptr;
        }

        for (unsigned int atom_number : atom_numbers) {
            if (PyList_Append(atom_numbers_list, Py_BuildValue("i", atom_number))) {
                Py_DecRef(atom_numbers_list);
                Py_DecRef(result_list);
                return nullptr;
            }
        }

        const char *bytes = reinterpret_cast<const char *>(result.array);
        PyObject *system = Py_BuildValue("y#O",
                                         bytes, result.array_size * sizeof(double), atom_numbers_list);
        if (system == nullptr) {
            Py_DecRef(atom_numbers_list);
            Py_DecRef(result_list);
            return nullptr;
        }

        if (PyList_Append(result_list, system)) {
            Py_DecRef(system);
            Py_DecRef(result_list);
            return nullptr;
        }
    }

    return result_list;
}

static PyObject *fit(PyObject *, PyObject *args) {
    PresetInfo info;
    if (!ParseArgs(args, info)) {
        return nullptr;
    }

    std::pair<double, double> fitted_range;
    if (!PresetsFitMBTR(info, fitted_range)) {
        return nullptr;
    }

    return Py_BuildValue("dd", fitted_range.first, fitted_range.second);
}

static PyMethodDef MBTR_Methods[] = {
        {"compute", (PyCFunction) compute, METH_VARARGS, nullptr},
        {"fit",     (PyCFunction) fit,     METH_VARARGS, nullptr},
        {nullptr,   nullptr}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "mbtr_imp",
        nullptr,
        0,
        MBTR_Methods,
        nullptr,
        nullptr,
        nullptr,
        nullptr
};
#define INITERROR return NULL

PyMODINIT_FUNC PyInit_mbtr_imp()
#else
#define INITERROR return
void initmbtr_imp()
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&module_def);
#else
    PyObject *module = Py_InitModule("mbtr_imp", MBTR_Methods);
#endif

    if (module == nullptr)
        INITERROR;

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
