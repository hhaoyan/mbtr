import logging

import matplotlib.pyplot as plt
import numpy
import scipy.linalg
import scipy.sparse
import scipy.spatial
from sklearn.metrics.pairwise import euclidean_distances


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

        # train model
        k = self.kernel.get_kernel_matrix(training_data[0], training_data[0])
        u = scipy.linalg.cholesky(k + self.par_lambda * numpy.identity(k.shape[0]))
        beta = scipy.linalg.solve_triangular(u, training_data[1], trans='T')
        alpha = scipy.linalg.solve_triangular(u, beta)

        self.alpha = alpha.reshape(alpha.size, 1)
        self.u = u

    def predict(self, input_data):
        """
        Predicts the y according to input_data.
        FIXME: array shape checking.

        Args:
            input_data (numpy.ndarray): x_i^j.

        Returns:
            (numpy.ndarray, numpy.ndarray): y and variance
        """
        l = self.kernel.get_kernel_matrix(self.training_data[0], input_data)
        y = numpy.dot(l.T, self.alpha)

        beta = scipy.linalg.solve_triangular(self.u, l, trans='T')
        c = scipy.linalg.solve_triangular(self.u, beta)
        v = self.kernel.get_kernel_diagonal(input_data) - numpy.sum(l * c, axis=0)
        return numpy.asarray(y).flatten(), numpy.asarray(v).reshape((v.size, 1))

    def validate(self, input_data):
        """
        Validate according to the validation data. Giving MSE, MAE, R2.

        Args:
             input_data ((numpy.ndarray, numpy.ndarray)): x_i^j, y_i.
        """
        y = input_data[1]
        hat_y, variance = self.predict(input_data[0])

        plt.scatter(hat_y, input_data[1], s=1)
        plt.show()

        rmse = numpy.sqrt(numpy.mean((hat_y - y) ** 2))
        mae = numpy.mean(numpy.absolute(hat_y - y))

        N = hat_y.size
        cross_mean = numpy.mean(hat_y * y)
        y_mean = numpy.mean(y)
        hat_y_mean = numpy.mean(hat_y)
        y_2_mean = numpy.mean(y ** 2)
        hat_y_2_mean = numpy.mean(hat_y ** 2)
        r_squared = (N * cross_mean - hat_y_mean * y_mean) ** 2 / (N * hat_y_2_mean - hat_y_mean ** 2) / (
                N * y_2_mean - y_mean ** 2)

        return rmse, mae, r_squared


class AutoTrainTwoParam(object):
    def __init__(self, data, kernel, sigma_start, noise_level_start):
        self.data = data
        self.kernel = kernel(sigma_start)

        self._sigma_changed = None

        self.sigma = sigma_start
        self.noise_level = noise_level_start
        self.performance = None
        self.krr = None

    def _evaluate_model(self):
        self.kernel.par_sigma = self.sigma
        self.krr = KernelRidgeRegression(self.data[0], self.kernel, self.sigma, self.noise_level)

        rmse, mae, r2 = self.krr.validate(self.data[1])
        logging.debug('Evaluate model: sigma: %.5e, noise level %.5e, RMSE: %.6f, MAE: %.6f, R2: %.6f',
                      self.sigma, self.noise_level, rmse, mae, r2)

        self.performance = rmse, mae, r2

    def _train_sigma(self):
        old_performance = self.performance

        self.sigma *= 1.05
        self._evaluate_with_linalg_error()
        self.sigma /= 1.05
        performance_plus_5p = self.performance
        self.sigma *= 0.95
        self._evaluate_with_linalg_error()
        self.sigma /= 0.95
        performance_minus_5p = self.performance

        logging.debug('Now %f, %f, %f at sigma = %f',
                      performance_minus_5p[0], old_performance[0], performance_plus_5p[0], self.sigma)
        if performance_plus_5p[0] > old_performance[0] > performance_minus_5p[0]:
            self._sigma_changed = True
            # should decrease sigma
            for level in [0.5, 0.90, 0.65, 0.85, 0.75, 0.8]:
                self.sigma *= level
                self._evaluate_with_linalg_error()
                performance = self.performance
                if performance[0] < performance_minus_5p[0]:
                    logging.debug('Select down - %f.', level)
                    self._train_sigma()
                    return
                self.sigma /= level

            logging.debug('sigma *= 0.95 will be fine...')
            self.sigma *= 0.95
            self.performance = performance_minus_5p
            return
        elif performance_plus_5p[0] < old_performance[0] < performance_minus_5p[0]:
            self._sigma_changed = True
            for level in [2.0, 1.1, 1.75, 1.15, 1.5, 1.20, 1.35]:
                self.sigma *= level
                self._evaluate_with_linalg_error()
                if self.performance[0] < performance_plus_5p[0]:
                    logging.debug('Select up - %f.', level)
                    self._train_sigma()
                    return
                self.sigma /= level

            logging.debug('sigma *= 1.05 will be fine...')
            self.sigma *= 1.05
            self.performance = performance_plus_5p
            return
        else:
            self.performance = old_performance
            return

    def _evaluate_with_linalg_error(self):
        try:
            self._evaluate_model()
            return True
        except (numpy.linalg.LinAlgError, scipy.linalg.LinAlgError):
            logging.debug('Oops, LinAlgError... at lambda = %.5f', self.noise_level)
            self.performance = 9e99, 9e99, 0.0
            return False

    def _train_noise_level(self):
        old_performance = self.performance

        self.noise_level *= 2
        self._evaluate_with_linalg_error()
        self.noise_level /= 2
        performance_plus = self.performance
        self.noise_level *= 0.5
        self._evaluate_with_linalg_error()
        self.noise_level /= 0.5
        performance_minus = self.performance

        if performance_plus[0] > old_performance[0] > performance_minus[0]:
            self.noise_level *= 0.5
            self.performance = performance_minus
            self._train_noise_level()
            return
        elif performance_plus[0] < old_performance[0] < performance_minus[0]:
            self.noise_level *= 2
            self.performance = performance_plus
            self._train_noise_level()
            return
        else:
            self.performance = old_performance
            return

    def auto_train(self):
        self._evaluate_with_linalg_error()
        while True:
            self._train_sigma()
            if not self._sigma_changed:
                break
            self._sigma_changed = False
            self._train_noise_level()

        self._evaluate_model()
