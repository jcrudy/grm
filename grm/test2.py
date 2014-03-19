import numpy
from grm2 import GeneralizedRegressor
import scipy.stats


m = 1000
n = 10
class TestGeneralizedLinearRegressor(object):
    def __init__(self):
        numpy.random.seed(1)
        self.X = numpy.random.normal(size=(m,n))
        self.beta = numpy.random.binomial(1,.5,size=n) * numpy.random.uniform(.5,2.5,size=n)
        self.eta = numpy.dot(self.X, self.beta)
        
    def test_gaussian(self):
        model = GeneralizedRegressor()
        mu = self.eta
        model.fit(self.X, mu)
        assert scipy.stats.pearsonr(model.regressor_.coef_, self.beta) > .9

    def test_binomial(self):
        model = GeneralizedRegressor()
        mu = 1.0 / (1.0 + numpy.exp(-self.eta))
        model.fit(self.X, mu)
        assert scipy.stats.pearsonr(model.regressor_.coef_, self.beta) > .9
    
    
    
    
if __name__ == '__main__':
    import nose
    nose.run(argv=[__file__, '-s', '-v'])