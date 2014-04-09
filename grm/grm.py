from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy
import scipy.stats as stats
from abc import ABCMeta, abstractmethod

class LinearRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = numpy.ones_like(y)
        self.intercept_ = numpy.average(y, weights=sample_weight)
        y_ = (y - self.intercept_)*numpy.sqrt(sample_weight)
        X_ = numpy.dot(numpy.diag(numpy.sqrt(sample_weight)),X)
        self.coef_ = numpy.linalg.lstsq(X_, y_)[0]
        return self

    def predict(self, X):
        return self.intercept_ + numpy.dot(X, self.coef_)


class LossFunction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, mu, y, **kwargs):
        '''
        Subclasses should implement this method to return a tuple containing the
        new target and the new sample_weight.  The latter may be a vector if the
        matrix would be diagonal.
        '''

    @abstractmethod
    def predict(self, eta, **kwargs):
        '''
        Subclasses should implement this method to return a prediction mu from
        the output, eta, of the inner regression model.  Essentially, this
        method should implement the inverse of the link function.
        '''

    @abstractmethod
    def get_params(self, kwargs):
        '''
        All this does is pick out the relevant arguments from kwargs and return
        a dict containing only them.
        '''

    @abstractmethod
    def eval(self, mu, y, **kwargs):
        '''
        Evaluate the loss function
        '''

    @abstractmethod
    def inverse_predict(self, mu, **kwargs):
        '''
        Compute eta from mu
        '''

    @abstractmethod
    def starting_point(self, X, y, base_regressor, loss_function, **kwargs):
        pass

    @abstractmethod
    def alter_arguments(self, X, y, kwargs):
        pass

class LinkableLossFunction(LossFunction):
    def __init__(self, link_function):
        self.link_function = link_function

    def predict(self, eta, **kwargs):
        link_params = self.link_function.get_params(kwargs)
        return self.link_function.inv(eta, **link_params)

    def inverse_predict(self, mu, **kwargs):
        link_params = self.link_function.get_params(kwargs)
        return self.link_function.eval(mu, **link_params)

    def alter_arguments(self, X, y, kwargs):
        return X, y, kwargs

class IndependentLinkableLossFunction(LinkableLossFunction):

    @abstractmethod
    def hessian(self, mu, y, **kwargs):
        '''
        Compute the diagonal of the Hessian
        '''
        pass

    @abstractmethod
    def score(self, mu, y, **kwargs):
        '''
        Compute the score vector
        '''
        pass

    def step(self, mu, y, **kwargs):
        link_params = self.link_function.get_params(kwargs)
        loss_params = self.get_params(kwargs)
        eta = self.link_function.eval(mu, **link_params)
        hessian = self.hessian(mu, y, **loss_params)
        score = self.score(mu, y, **loss_params)
        link_deriv = self.link_function.deriv(mu, **link_params)

        w = 0.5 * hessian
        w *= link_deriv * link_deriv
        z = eta - 0.5 * (1.0 / w) * score * (1.0 / link_deriv)
        return z, w

class NormalLossFunction(IndependentLinkableLossFunction):
    def get_params(self, kwargs):
        return {}

    def hessian(self, mu, y):
        m = mu.shape[0]
        return 2 * numpy.ones(m, dtype = mu.dtype)

    def score(self, mu, y):
        return 2 * (mu - y)

    def eval(self, mu, y):
        return numpy.sum((y - mu) ** 2)

    def starting_point(self, X, y, base_regressor, loss_function, **kwargs):
        return y

class BinomialLossFunction(IndependentLinkableLossFunction):
    def __init__(self, link_function, starting_point_factor = 0.1):
        super(BinomialLossFunction, self).__init__(link_function)
        self.starting_point_factor = starting_point_factor

    def get_params(self, kwargs):
        return {'n': kwargs['n']}

    def hessian(self, mu, y, n):
        return (mu * mu * n + y + 2 * mu * y) / (mu * mu * ((1.0 - mu)**2))

    def score(self, mu, y, n):
        return (n * mu - y) / (mu * (1.0 - mu))

    def eval(self, mu, y, n):
        return 2 * numpy.sum(y[y>0]*numpy.log(y[y>0]/mu[y>0])) + \
               2 * numpy.sum((n[y<n]-y[y<n])*numpy.log((n[y<n]-y[y<n])/(n[y<n]-mu[y<n])))
        #
        # one_minus_mu = 1.0 - mu
        # y_over_zero = y > 0
        # return -numpy.sum(y[y_over_zero] * numpy.log(mu[y_over_zero] /
        #                   one_minus_mu[y_over_zero])) - numpy.sum(
        #                   n * numpy.log(one_minus_mu))

    def starting_point(self, X, y, base_regressor, loss_function, n):
        mean = numpy.mean(y / n)
        return self.starting_point_factor * mean + \
               (1.0 - self.starting_point_factor) * y / n

class LogHazardLossFunction(LossFunction):
    def get_params(self, kwargs):
        return {'c': kwargs['c'],
                'N': kwargs['N'],
                'nu': kwargs['nu'],
                'rank': kwargs['rank']}

    def alter_arguments(self, X, y, kwargs):
        # Compute sort-based extra arguments
        m = y.shape[0]
        order = numpy.argsort(y)
        rank = stats.rankdata(y).astype(int) - 1
        N = m - rank
        nu = y
        nu[rank > 0] -= y[order][rank[rank > 0] - 1]
        kwargs = kwargs.copy()
        kwargs['rank'] = rank
        kwargs['nu'] = nu
        kwargs['N'] = N
        return numpy.c_[X, y], y, kwargs

    def eval(self, mu, y, c, N, nu, rank):
        return -numpy.sum(c * mu) + numpy.sum(nu * numpy.exp(mu) * N)

    def predict(self, eta, c, N, nu, rank):
        return eta

    def inverse_predcit(self, mu, c, N, nu, rank):
        return mu

    def starting_point(self, X, y, base_regressor, loss_function,
                       c, N, nu, rank):
        return numpy.log(c / (nu * N))

    def step(self, mu, y, c, N, nu, rank):
        w = 0.5 * nu * numpy.exp(mu) * N
        z = mu + (0.5 * c / w) - 1.0
        return z, w

# class ExponentialFamilyLossFunction(LinkableLossFunction):
#     @abstractmethod
#     def variance(self, mu, y, **kwargs):
#         '''
#         Subclasses should implement this method to return the value of the
#         variance function for this distribution at this mu.
#         '''
#
#     def step(self, eta, y, **kwargs):
#         '''
#         This implementation uses Fisher scoring instead of an actual Newton
#         step, which is pretty standard.
#         '''
#         link_params = self.link_function.get_params(kwargs)
#         loss_params = self.get_params(kwargs)
#         variance = self.variance(mu, **loss_params)
#         link_deriv = self.link_function.deriv(mu, **link_params)
#
#         w = 1.0 / ((link_deriv ** 2) * variance)
#         z = eta + (y - mu) * link_deriv
#         return z, w
#
# class BinomialLossFunction(ExponentialFamilyLossFunction):
#     def variance(self, mu, y, n_trials):
#         return n_trials * mu * (1.0 - mu)
#
#     def get_params(self, kwargs):
#         return {'n_trials': kwargs['n_trials']}
#
# class NormalLossFunction(ExponentialFamilyLossFunction):
#     def get_params(self, kwargs):
#         return {}
#
#     def variance(self, mu, y):
#         return 1.0

class LinkFunction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_params(self, kwargs):
        pass

    @abstractmethod
    def eval(self, mu):
        pass

    @abstractmethod
    def deriv(self, mu):
        pass

    @abstractmethod
    def inv(self, eta):
        pass

class IdentityLink(LinkFunction):
    def get_params(self, kwargs):
        return {}

    def eval(self, mu):
        return mu

    def inv(self, eta):
        return eta

    def deriv(self, mu):
        return numpy.ones_like(mu)

class LogitLink(LinkFunction):
    def get_params(self, kwargs):
        return {}

    def eval(self, mu):
        return -numpy.log(1.0 / 1.0 - mu)

    def inv(self, eta):
        return 1.0 / (1.0 + numpy.exp(-eta))

    def deriv(self, mu):
        return 1.0 / (mu - mu**2)

class ExpLink(LinkFunction):
    def get_params(self, kwargs):
        return {}

    def eval(self, mu):
        return numpy.exp(mu)

    def inv(self, eta):
        return numpy.log(eta)

    def deriv(self, mu):
        return numpy.exp(mu)

class Shrinker(object):
    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, X, y, base_regressor, loss_function, **kwargs):
        mean = numpy.mean(y)
        return self.factor*mean + (1.0 - self.factor)*y

class ConvergenceTester(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def append(self, regressor, loss, X, y, eta, mu):
        pass

    @abstractmethod
    def check(self):
        pass

class StableStatisticConvergenceTester(ConvergenceTester):
    def __init__(self, n=2, threshold=.0001):
        self.n = n
        self.threshold = threshold
        self.stats = []

    @abstractmethod
    def statistic(self, regressor, loss, X, y, eta, mu):
        pass

    def append(self, regressor, loss, X, y, eta, mu):
        self.stats.append(self.statistic(regressor, loss, X, y, eta, mu))

    def check(self):
        n = self.n
        if len(self.stats) < n:
            return False
        else:
            for i in range(2, n + 1):
                if numpy.sum(numpy.abs(self.stats[1-i] - self.stats[-i])) \
                    >= self.threshold:
                    return False
        return True

class StableLossConvergenceTester(StableStatisticConvergenceTester):
    def statistic(self, regressor, loss, X, y, eta, mu):
        return loss

class StablePredictionConvergenceTester(StableStatisticConvergenceTester):
    def statistic(self, regressor, loss, X, y, eta, mu):
        return mu / float(len(mu))

class GeneralizedRegressor(BaseEstimator):
    def __init__(self, base_regressor=LinearRegressor(),
        loss_function=NormalLossFunction(IdentityLink()),
        max_iter=20, convergence_test=StableLossConvergenceTester()):
        self.base_regressor = base_regressor
        self.loss_function = loss_function
        self.max_iter = max_iter
        self.convergence_test = convergence_test

    def fit(self, X, y, **kwargs):
        # Alter arguments if necessary
        X, y, kwargs = self.loss_function.alter_arguments(X, y, kwargs)

        # Initialize mu, eta, and loss before first iteration
        mu = self.loss_function.starting_point(X, y, self.base_regressor,
                                               self.loss_function, **kwargs)
        eta = self.loss_function.inverse_predict(mu, **kwargs)
        loss = self.loss_function.eval(mu, y, **kwargs)
        relaxation_factor = 1.0
        i = 0
        while i <= self.max_iter:

            # Compute the adjusted response and weights for the inner fitting step
            z, w = self.loss_function.step(mu, y, **kwargs)

            # Fit the new inner regressor
            new_regressor = clone(self.base_regressor)
            new_regressor.fit(X, z, sample_weight=w)

            # Estimate the new eta and mu
            new_eta = new_regressor.predict(X)
            new_mu = self.loss_function.predict(new_eta, **kwargs)

            # Calculate the loss
            new_loss = self.loss_function.eval(new_mu, y, **kwargs)

            # Check for convergence
            self.convergence_test.append(new_regressor, new_loss, X, y, new_eta, new_mu)
            if self.convergence_test.check():
                regressor = new_regressor
                break

            # Adjust the relaxation factor for the next iteration
            if new_loss > loss:
                relaxation_factor *= 0.5
            elif new_loss < loss and relaxation_factor < 1.0:
                relaxation_factor = 1.0

            # Apply the relaxation factor for the next iteration
            if relaxation_factor != 1.0:
                new_eta = (relaxation_factor)*eta + (1.0 - relaxation_factor)*new_eta
                new_mu = self.loss_function.predict(new_eta, **kwargs)

            # Prepare for the next iteration
            eta = new_eta
            mu = new_mu
            loss = new_loss
            regressor = new_regressor
            i += 1

        self.regressor_ = regressor
        return self

    def predict(self, X, **kwargs):
        # Predict from the inner regressor
        eta = self.regressor_.predict(X)

        # Apply the loss function's predict method to get the final prediction
        return self.loss_function.predict(eta, **kwargs)
