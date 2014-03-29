import numpy
from grm2 import GeneralizedRegressor, BinomialLoss, LogitLink, LinearRegressor
import scipy.stats
from pyearth.earth import Earth


m = 1000
n = 10
class GeneralizedRegressorTester(object):
    def test_gaussian(self):
        model = GeneralizedRegressor(base_regressor=self.base_regressor())
        mu = self.eta
        model.fit(self.X, mu)
        assert self.assertion(model)

    def test_binomial(self):
        model = GeneralizedRegressor(base_regressor=self.base_regressor(), loss_function=BinomialLoss(1), link_function=LogitLink(1))
        mu = 1.0 / (1.0 + numpy.exp(-self.eta))
        y = numpy.random.binomial(1, mu)
        model.fit(self.X, y)
        assert self.assertion(model)
        
#     def test_proportional_hazards(self):
#         model = GeneralizedRegressor(base_regressor=self.base_regressor(), loss_function=ProportionalHazardsLoss(), link_function=ExpLink())
#         mu = numpy.exp(self.eta)
#         
#         # Generate event times (time until event / death / failure)
#         T = numpy.random.exponential(mu, size=m)
#         
#         # Generate observation times (time until censoring)
#         obs = numpy.random.uniform(0., 365.25, size=m)
#         
#         # Calculate time and censor status.  c=1 indicates event, c=0 indicates right censor
#         c = 1.0 * (T <= obs)
#         y = numpy.minimum(T, obs)
#         model.fit(self.X, y, event=c)
        
        
    def assertion(self, model):
        return scipy.stats.pearsonr(model.predict(self.X), self.eta) > .99

class TestGeneralizedLinearRegressor(GeneralizedRegressorTester):
    base_regressor = LinearRegressor
    def __init__(self):
        numpy.random.seed(1)
        self.X = numpy.random.normal(size=(m,n))
        self.beta = numpy.random.binomial(1,.5,size=n) * numpy.random.uniform(.5,2.5,size=n)
        self.eta = numpy.dot(self.X, self.beta)
    
p = 10
def earth_basis(X, vars, parents, knots, signs):
    p = vars.shape[0]
    B = numpy.empty(shape=(m,p+1))
    B[:,0] = 1.0
    for i in range(p):
        knot = numpy.sort(X[:,vars[i]])[knots[i]]
        B[:,i+1] = B[:,parents[i]] * numpy.maximum(signs[i]*(X[:,vars[i]] - knot), 0.0)
    return B
        
class TestGeneralizedEarthRegressor(GeneralizedRegressorTester):
    base_regressor = Earth
    def __init__(self):
        numpy.random.seed(1)
        self.X = numpy.random.normal(size=(m,n))
        self.vars = numpy.argmax(numpy.random.multinomial(1, (1.0/float(n))*numpy.ones(n), p),1)
        self.knots = numpy.random.randint(6, m-6, size=p)
        self.parents = numpy.array([numpy.random.binomial(i, 1.0/float(p**2)) if i>0 else 0 for i in range(p)])
        self.signs = numpy.random.binomial(1, .5, size=p)
        self.B = earth_basis(self.X, self.vars, self.parents, self.knots, self.signs)
        self.beta = numpy.random.uniform(-2.0,2.0,size=p+1)
        self.eta = numpy.dot(self.B, self.beta)
    
    
if __name__ == '__main__':
    import nose
    nose.run(argv=[__file__, '-s', '-v'])