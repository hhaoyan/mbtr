import functools
import logging

import numpy
import scipy.linalg
import scipy.sparse
import scipy.spatial
import sklearn.metrics
from sklearn.metrics.pairwise import euclidean_distances

__author__ = "Haoyan Huo"
__maintainer__ = "Haoyan Huo"
__email__ = "haoyan.huo@lbl.gov"


class Kernel(object):
    def get_kernel_matrix(self, x1, x2):
        raise NotImplementedError()

    def get_kernel_diagonal(self, x):
        k = numpy.empty((x.shape[0],))
        k.fill(1.0)
        return k


class GaussianKernel(Kernel):
    def __init__(self, par_sigma):
        """
        A Gaussian kernel generator.

        Args:
             par_sigma (float): sigma.
        """
        self.par_sigma = par_sigma

        self._cached_norm_matrix = {}

    def get_kernel_matrix(self, x1, x2):
        """
        Get the kernel K(x1, x2). Shapes like (N, d), (M, d). Returns (N, M).

        Args:
            x1 (numpy.ndarray): x_i^j.
            x2 (numpy.ndarray): x_i^j.
        """
        hash_value = id(x1) + id(x2)
        if hash_value not in self._cached_norm_matrix:
            k = euclidean_distances(x1, x2, squared=True)
            self._cached_norm_matrix[hash_value] = k

        return numpy.exp(self._cached_norm_matrix[hash_value] * (-0.5 / (self.par_sigma ** 2)))


class LaplacianKernel(GaussianKernel):
    def get_kernel_matrix(self, x1, x2):
        hash_value = id(x1) + id(x2)
        if hash_value not in self._cached_norm_matrix:
            if scipy.sparse.issparse(x1) or scipy.sparse.issparse(x2):
                raise ValueError('Sorry, no sparse implementation for p-1 norm.')
            k = scipy.spatial.distance.cdist(x1, x2, 'minkowski', 1)
            self._cached_norm_matrix[hash_value] = k

        return numpy.exp(self._cached_norm_matrix[hash_value] * (-1 / self.par_sigma))


class KernelRidgeRegression(object):
    def __init__(self, training_data, kernel, par_sigma, par_lambda):
        """
        Does Kernel ridge regression.

        Args:
             training_data((numpy.ndarray, numpy.ndarray)): x_i^j, y_i
             kernel (Kernel): a kernel.
             par_sigma (float): parameter sigma.
             par_lambda (float): parameter lambda.
        """
        self.kernel = kernel
        self.par_sigma = par_sigma
        self.par_lambda = par_lambda
        self.training_data = training_data

        self.assert_data_shape(training_data)

        # train model
        k = self.kernel.get_kernel_matrix(training_data[0], training_data[0])
        u = scipy.linalg.cholesky(k + self.par_lambda * numpy.identity(k.shape[0]))
        beta = scipy.linalg.solve_triangular(u, training_data[1], trans='T')
        alpha = scipy.linalg.solve_triangular(u, beta)

        self.alpha = alpha.reshape(alpha.size, 1)
        self.u = u

    @staticmethod
    def assert_data_shape(data):
        assert isinstance(data, tuple)
        assert len(data) == 2
        assert len(data[0].shape) == 2
        assert len(data[1].shape) == 1
        assert data[0].shape[0] == data[1].shape[0]

    def predict(self, input_data):
        """
        Predicts the y according to input_data.

        Args:
            input_data (numpy.ndarray): x_i^j.

        Returns:
            (numpy.ndarray, numpy.ndarray): y and variance
        """
        assert len(input_data.shape) == 2
        assert input_data.shape[1] == self.training_data[0].shape[1]

        l = self.kernel.get_kernel_matrix(self.training_data[0], input_data)
        y = numpy.dot(l.T, self.alpha)

        beta = scipy.linalg.solve_triangular(self.u, l, trans='T')
        c = scipy.linalg.solve_triangular(self.u, beta)
        v = self.kernel.get_kernel_diagonal(input_data) - numpy.sum(l * c, axis=0)
        return numpy.asarray(y).flatten(), numpy.asarray(v).reshape((v.size, 1))

    @staticmethod
    def stat(y, hat_y):
        rmse = numpy.sqrt(sklearn.metrics.mean_squared_error(y, hat_y))
        mae = sklearn.metrics.mean_absolute_error(y, hat_y)
        r2 = sklearn.metrics.r2_score(y, hat_y)
        return rmse, mae, r2

    def validate(self, input_data):
        """
        Validate according to the validation data. Giving MSE, MAE, R2.

        Args:
             input_data ((numpy.ndarray, numpy.ndarray)): x_i^j, y_i.
        """
        self.assert_data_shape(input_data)

        y = input_data[1]
        hat_y, variance = self.predict(input_data[0])

        # plt.scatter(hat_y, input_data[1], s=1)
        # plt.show()

        return self.stat(y, hat_y)


class AutoTrainTwoParam(object):
    def __init__(self, data, kernel, sigma_start, noise_level_start):
        self.data = data
        self.kernel = kernel

        self._sigma_changed = None

        self.sigma = sigma_start
        self.noise_level = noise_level_start

    @property
    def krr(self):
        kernel = self.kernel(self.sigma)
        return KernelRidgeRegression(self.data[0], kernel, self.sigma, self.noise_level)

    @functools.lru_cache(maxsize=128, typed=False)
    def _evaluate_model(self, sigma, noise_level):
        try:
            kernel = self.kernel(sigma)
            krr = KernelRidgeRegression(self.data[0], kernel, sigma, noise_level)

            rmse, mae, r2 = krr.validate(self.data[1])
            logging.debug('(σ: %.5e, λ: %.5e) ==> RMSE: %.6f, MAE: %.6f, R2: %.6f',
                          sigma, noise_level, rmse, mae, r2)

            return rmse, mae, r2
        except (numpy.linalg.LinAlgError, scipy.linalg.LinAlgError):
            logging.debug('(σ: %.5e, λ: %.5e) ==> LinAlgError',
                          sigma, noise_level)
            return 9e99, 9e99, 0.0

    def _train_sigma(self):
        old_performance = self.current_performance()

        self.sigma *= 1.05
        performance_plus_5p = self.current_performance()
        self.sigma /= 1.05

        self.sigma *= 0.95
        performance_minus_5p = self.current_performance()
        self.sigma /= 0.95

        logging.debug('Now %f, %f, %f at σ = %f',
                      performance_minus_5p[0], old_performance[0], performance_plus_5p[0], self.sigma)
        if performance_plus_5p[0] > old_performance[0] > performance_minus_5p[0]:
            self._sigma_changed = True
            # should decrease sigma
            for level in [0.5, 0.90, 0.65, 0.85, 0.75, 0.8]:
                self.sigma *= level
                performance = self.current_performance()
                if performance[0] < performance_minus_5p[0]:
                    logging.debug('Select down - %f.', level)
                    self._train_sigma()
                    return
                self.sigma /= level

            logging.debug('σ *= 0.95 will be fine.')
            self.sigma *= 0.95
            return
        elif performance_plus_5p[0] < old_performance[0] < performance_minus_5p[0]:
            self._sigma_changed = True
            for level in [2.0, 1.1, 1.75, 1.15, 1.5, 1.20, 1.35]:
                self.sigma *= level
                performance = self.current_performance()
                if performance[0] < performance_plus_5p[0]:
                    logging.debug('Select up - %f.', level)
                    self._train_sigma()
                    return
                self.sigma /= level

            logging.debug('σ *= 1.05 will be fine.')
            self.sigma *= 1.05
            return
        else:
            return

    def current_performance(self):
        return self._evaluate_model(self.sigma, self.noise_level)

    def _train_noise_level(self):
        old_performance = self.current_performance()

        self.noise_level *= 2
        performance_plus = self.current_performance()
        self.noise_level /= 2

        self.noise_level *= 0.5
        performance_minus = self.current_performance()
        self.noise_level /= 0.5

        if performance_plus[0] > old_performance[0] > performance_minus[0]:
            self.noise_level *= 0.5
            self._train_noise_level()
            return
        elif performance_plus[0] < old_performance[0] < performance_minus[0]:
            self.noise_level *= 2
            self._train_noise_level()
            return
        else:
            return

    def auto_train(self):
        while True:
            self._train_sigma()
            if not self._sigma_changed:
                break
            self._sigma_changed = False
            self._train_noise_level()
