from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy
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

class Function(object):
    __metaclass__ = ABCMeta
    param_names = {}
    def extract_params(self, kwargs):
        return dict([(k, kwargs[k]) for k in kwargs.iterkeys() if k in self.param_names])
    
    @abstractmethod
    def eval(self, y, mu, **kwargs):
        pass
    
class LossFunction(Function):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def deriv2(self, y, mu, **kwargs):
        pass
    
class SquaredErrorLoss(LossFunction):
    def eval(self, y, mu):
        return numpy.sum((y - mu)**2)
    
    def deriv2(self, y, mu):
        return 2.0 * numpy.ones_like(mu)
    
class BinomialLoss(LossFunction):
    def __init__(self, m):
        self.m = m
        
    def eval(self, y, mu):
        return 2 * numpy.sum(y[y>0]*numpy.log(y[y>0]/mu[y>0])) + 2 * numpy.sum((self.m-y[y<self.m])*numpy.log((self.m-y[y<self.m])/(self.m-mu[y<self.m])))

    def deriv2(self, y, mu):
        return self.m*mu * (self.m - self.m*mu)

class LinkFunction(Function):
    __metaclass__ = ABCMeta

    @abstractmethod
    def inv(self, eta, **kwargs):
        pass
    
    @abstractmethod
    def deriv(self, mu, **kwargs):
        pass
    
class IdentityLink(LinkFunction):
    def eval(self, mu):
        return mu
    
    def inv(self, eta):
        return eta
    
    def deriv(self, mu):
        return numpy.ones_like(mu)

class LogitLink(LinkFunction):
    def eval(self, mu):
        return -numpy.log(self.m/mu - 1.0)
    
    def inv(self, eta):
        return self.m / (1.0 + numpy.exp(-eta))
    
    def deriv(self, mu):
        return 1.0 / (self.m*(mu - mu**2))
        
class Shrinker(object):
    def __init__(self, factor=0.1):
        self.factor = factor
        
    def __call__(self, y):
        mean = numpy.mean(y)
        return self.factor*mean + (1.0 - self.factor)*y
    

class GeneralizedRegressor(BaseEstimator):
    def __init__(self, base_regressor=LinearRegressor(), loss_function=SquaredErrorLoss(), 
                 link_function=IdentityLink(), starter=Shrinker(0.1), max_iter=20, 
                 convergence_threshold=1e-8):
        self.base_regressor = base_regressor
        self.loss_function = loss_function
        self.link_function = link_function
        self.starter = starter
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold

    def fit(self, X, y, **kwargs):
        # Get any additional parameters, such as exposure, from the arguments
        link_params = self.link_function.extract_params(kwargs)
        loss_params = self.loss_function.extract_params(kwargs)
        
        # Initialize mu, eta, and loss before first iteration
        mu = self.starter(y)
        eta = self.link_function.eval(mu, **link_params)
        loss = float('inf')
        relaxation_factor = 1.0
        i = 0
        while i <= self.max_iter:
            
            # Compute the adjusted response and weights for the inner fitting step
            z = eta + (y - mu)*self.link_function.deriv(mu, **link_params)
            w = 1.0 / ((self.link_function.deriv(mu, **link_params)**2)*self.loss_function.deriv2(y, mu, **loss_params))
            
            # Fit the new inner regressor
            new_regressor = clone(self.base_regressor).fit(X, z, sample_weight=w)
            
            # Estimate the new eta and mu
            new_eta = new_regressor.predict(X)
            new_mu = self.link_function.inv(new_eta, **link_params)
            
            # Check for convergence
            denominator = numpy.average(eta**2, weights=w)
            numerator = numpy.average((eta - new_eta)**2, weights=w)
            if numerator / float(denominator) < self.convergence_threshold:
                self.regressor_ = new_regressor
                break
            
            # Calculate the loss
            new_loss = self.loss_function.eval(y, new_mu, **loss_params)
            
            # Adjust the relaxation factor for the next iteration
            if new_loss > loss:
                relaxation_factor *= 0.5
            elif new_loss < loss and relaxation_factor < 1.0:
                relaxation_factor = 1.0
            
            # Apply the relaxation factor for the next iteration
            if relaxation_factor != 1.0:
                new_eta = (relaxation_factor)*eta + (1.0 - relaxation_factor)*new_eta
                new_mu = self.link_function.inv(new_eta, **link_params)
            
            # Prepare for the next iteration
            eta = new_eta
            mu = new_mu
            loss = new_loss
            regressor = new_regressor
            i += 1
        
        self.regressor_ = regressor
        return self
    
    def predict(self, X, **kwargs):
        # Get any additional parameters, such as exposure, from the arguments
        link_params = self.link_function.extract_params(kwargs)
        
        # Predict from the inner regressor
        eta = self.regressor_.predict(X)
        
        # Apply the inverse link function to get the final prediction
        return self.link_function.inv(eta, **link_params)
        
        
        
        
        
