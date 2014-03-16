from sklearn.base import BaseEstimator, clone, RegressorMixin
import abc
import numpy
import warnings

class LinearRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = numpy.ones_like(y)
        self.intercept_ = numpy.average(y, weights=sample_weight)
        y_ = (y - self.intercept_)*numpy.sqrt(sample_weight)
        X_ = numpy.dot(numpy.diag(numpy.sqrt(sample_weight)),X)
        self.coef_ = numpy.linalg.lstsq(X_, y_)[0]
        
    def predict(self, X):
        return self.intercept_ + numpy.dot(X, self.coef_)
        
class GeneralizedRegressor(BaseEstimator):
    def __init__(self, base_regressor, grm_configuration, max_iter=10, convergence_threshold=1e-8):
        self.base_regressor = base_regressor
        self.grm_configuration = grm_configuration
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        
    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        link_params, likelihood_params = self.grm_configuration.take_params(*args, **kwargs)
        mu = self.grm_configuration.initialize_mu(X, y)
        self.record_ = []
        eta = self.grm_configuration.link(mu, link_params)
        i = 0
        score = float('inf')
        self.regressor_ = None
        converged = False
        while (not converged) and (i < self.max_iter):
            i += 1
            prev_regressor = self.regressor_
            self.regressor_ = clone(self.base_regressor)
            z = eta + (y - mu)*self.grm_configuration.link_deriv1(mu, link_params)
            w = -1.0 / ((self.grm_configuration.link_deriv1(mu, link_params)**2)*self.grm_configuration.likelihood_deriv2(mu, likelihood_params))
            self.regressor_.fit(X, z, sample_weight=w)
            prev_eta = eta
            prev_mu = mu
            eta = self.regressor_.predict(X, *args, **kwargs)
            mu = self.grm_configuration.link_inverse(eta, link_params)
            if not self.grm_configuration.link_domain.check(mu):
                warnings.warn('Link domain violated.  This may be due to complete separation of the training data or a similar issue.  The resulting model may not be reliable.')
                break
            prev_score = score
            score = self.grm_configuration.score(y, mu)
            if score >= prev_score:
                eta = prev_eta
                mu = prev_mu
                self.regressor_ = prev_regressor
                break
            self.record_.append(score)
            denominator = numpy.average(prev_eta**2, weights=w)
            numerator = numpy.average((prev_eta - eta)**2, weights=w)
            if numerator / float(denominator) < self.convergence_threshold:
                break
        return self
    
    def predict(self, X, *args, **kwargs):
        link_params, _ = self.grm_configuration.take_params(*args, **kwargs)
        eta = self.regressor_.predict(X)
        mu = self.grm_configuration.link_inverse(eta, link_params)
        return mu
    
    def score(self, X, y, *args, **kwargs):
        mu = self.predict(X, *args, **kwargs)
        return self.grm_configuration.score(y, mu)

class GrmConfiguration(object):
    @abc.abstractproperty
    def link_domain(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def link(self, mu, params):
        raise NotImplementedError
    
    @abc.abstractmethod
    def link_deriv1(self, mu, params):
        raise NotImplementedError
    
    @abc.abstractmethod
    def link_inverse(self, eta, params):
        raise NotImplementedError
    
    @abc.abstractmethod
    def likelihood_deriv2(self, mu, params):
        raise NotImplementedError
    
    @abc.abstractmethod
    def take_params(self, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def initialize_mu(self, X, y):
        raise NotImplementedError
    
    @abc.abstractmethod
    def score(self, y, mu):
        raise NotImplementedError
    
class ExponentialFamilyGrmConfiguration(GrmConfiguration):
    def __init__(self, link_function):
        self.approve_link_function(link_function)
        self.link_function = link_function
    
    @property
    def link_domain(self):
        return self.link_function.domain
    
    @abc.abstractmethod
    def approve_link_function(self, link_function):
        raise NotImplementedError
    
    def take_params(self, *args, **kwargs):
        return {}, {}
    
    def likelihood_deriv2(self, mu, params):
        return -self.variance_function(mu)
    
    def link(self, mu, params):
        return self.link_function.eval(mu, params)
    
    def link_deriv1(self, mu, params):
        return self.link_function.eval_deriv1(mu, params)
    
    def link_inverse(self, eta, params):
        return self.link_function.eval_inverse(eta, params)
    
    def initialize_mu(self, X, y):
        return y
    
    @abc.abstractmethod
    def variance_function(self, mu):
        raise NotImplementedError

class Interval(object):
    def __init__(self, lower=float('-inf'), upper=float('inf'), lower_closed=False, upper_closed=False):
        self.lower = lower
        self.upper = upper
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed
        
    def check(self, arr):
        res = arr < self.lower
        if not self.lower_closed:
            res = res | (arr == self.lower)
        res = res | (arr > self.upper)
        if not self.upper_closed:
            res = res | (arr == self.upper)
        return not numpy.any(res)

class LinkFunction(object):
    @abc.abstractmethod
    def eval(self, mu, params):
        raise NotImplementedError
    
    @abc.abstractmethod
    def eval_deriv1(self, mu, params):
        raise NotImplementedError
    
    @abc.abstractproperty
    def domain(self):
        raise NotImplementedError
    
class IdentityLinkFunction(LinkFunction):
    @property
    def domain(self):
        return Interval()
    
    def eval(self, mu, params):
        return mu 
    
    def eval_deriv1(self, mu, params):
        return numpy.ones_like(mu)
    
    def eval_inverse(self, eta, params):
        return eta

class LogitLinkFunction(LinkFunction):
    def __init__(self, m=1.0):
        self.m = float(m)
    
    @property
    def domain(self):
        return Interval(lower=0.0, upper=1.0)

    def eval(self, mu, params):
        return -numpy.log(self.m/mu - 1.0)
        
    def eval_deriv1(self, mu, params):
        return 1.0 / (self.m*(mu - mu**2))
    
    def eval_inverse(self, eta, params):
        return self.m / (1.0 + numpy.exp(-eta))
        

class NormalGrmConfiguration(ExponentialFamilyGrmConfiguration):
    def __init__(self):
        super(NormalGrmConfiguration, self).__init__(IdentityLinkFunction())
    
    def approve_link_function(self, link_function):
        assert isinstance(link_function, IdentityLinkFunction)
    
    def score(self, y, mu):
        return numpy.sum((y - mu)**2)
    
    def variance_function(self, mu):
        return numpy.ones_like(mu)
    

class BinomialGrmConfiguration(ExponentialFamilyGrmConfiguration):
    def __init__(self, m=1.0):
        self.m = float(m)
        super(BinomialGrmConfiguration, self).__init__(LogitLinkFunction(self.m))
        
    def initialize_mu(self, X, y):
        return 0.9*self.m*y + 0.1*self.m*(self.m - y)
    
    def approve_link_function(self, link_function):
        assert isinstance(link_function, LogitLinkFunction)
    
    def variance_function(self, mu):
        return self.m*mu * (self.m - self.m*mu)
    
    def score(self, y, mu):        
        return 2 * numpy.sum(y[y>0]*numpy.log(y[y>0]/mu[y>0])) + 2 * numpy.sum((self.m-y[y<self.m])*numpy.log((self.m-y[y<self.m])/(self.m-mu[y<self.m])))
        
      