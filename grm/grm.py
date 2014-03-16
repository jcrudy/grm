from sklearn.base import BaseEstimator, clone, RegressorMixin
import abc
import numpy

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
    def __init__(self, base_regressor, grm_configuration, max_iter=10):
        self.base_regressor = base_regressor
        self.grm_configuration = grm_configuration
        self.max_iter = max_iter
        
    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        
        link_params, likelihood_params = self.grm_configuration.take_params(*args, **kwargs)
        mu = self.grm_configuration.initialize_mu(X, y)
        self.record_ = []
        eta = self.grm_configuration.link(mu, link_params)
        i = 0
        while (not self.convergence_check()) and (i < self.max_iter):
            i += 1
            self.regressor_ = clone(self.base_regressor)
            z = eta + (y - mu)*self.grm_configuration.link_deriv1(mu, link_params)
            # FIXME: These weights seem to be wrong
            w = -1.0 / ((self.grm_configuration.link_deriv1(mu, link_params)**2)*self.grm_configuration.likelihood_deriv2(mu, likelihood_params))
            self.regressor_.fit(X, z, sample_weight=w)
            eta = self.regressor_.predict(X, *args, **kwargs)
            mu = self.grm_configuration.link_inverse(eta, link_params)
            self.record_.append(self.grm_configuration.score(y, mu))
        return self
    
    def convergence_check(self):
        return False
    
    def predict(self, X, *args, **kwargs):
        link_params, _ = self.grm_configuration.take_params(*args, **kwargs)
        eta = self.regressor_.predict(X)
        mu = self.grm_configuration.link_inverse(eta, link_params)
        return mu
    
    def score(self, X, y, *args, **kwargs):
        mu = self.predict(X, *args, **kwargs)
        return self.grm_configuration.score(y, mu)

class GrmConfiguration(object):
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

class LinkFunction(object):
    @abc.abstractmethod
    def eval(self, mu, params):
        raise NotImplementedError
    
    @abc.abstractmethod
    def eval_deriv1(self, mu, params):
        raise NotImplementedError
    
class IdentityLinkFunction(LinkFunction):
    def eval(self, mu, params):
        return mu 
    
    def eval_deriv1(self, mu, params):
        return numpy.ones_like(mu)
    
    def eval_inverse(self, eta, params):
        return eta

class LogitLinkFunction(LinkFunction):
    def __init__(self, m=1.0):
        self.m = float(m)

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
#         nonzero = y.copy()
#         nonzero[y==0] = 0.5
#         nonm = y.copy()
#         nonm[y==self.m]=0.5
        
        return 2 * numpy.sum(y[y>0]*numpy.log(y[y>0]/mu[y>0])) + 2 * numpy.sum((self.m-y[y<self.m])*numpy.log((self.m-y[y<self.m])/(self.m-mu[y<self.m])))
        

#         