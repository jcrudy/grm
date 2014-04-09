import numpy
from grm import GeneralizedRegressor, BinomialLossFunction, LogitLink
import scipy.stats
from pyearth.earth import Earth

numpy.seterr(all='raise')
m = 1000
n = 10
p = 10
def earth_basis(X, vars, parents, knots, signs):
    p = vars.shape[0]
    B = numpy.empty(shape=(m,p+1))
    B[:,0] = 1.0
    for i in range(p):
        knot = numpy.sort(X[:,vars[i]])[knots[i]]
        B[:,i+1] = B[:,parents[i]] * numpy.maximum(signs[i]*(X[:,vars[i]] - knot), 0.0)
    return B

numpy.random.seed(1)
X = numpy.random.normal(size=(m,n))
vars = numpy.argmax(numpy.random.multinomial(1, (1.0/float(n))*numpy.ones(n), p),1)
knots = numpy.random.randint(6, m-6, size=p)
parents = numpy.array([numpy.random.binomial(i, 1.0/float(p**2)) if i>0 else 0 for i in range(p)])
signs = numpy.random.binomial(1, .5, size=p)
B = earth_basis(X, vars, parents, knots, signs)
beta = numpy.random.uniform(-2.0,2.0,size=p+1)
eta = numpy.dot(B, beta)

model = GeneralizedRegressor(base_regressor=Earth(),
                             loss_function=BinomialLossFunction(LogitLink()))
n = numpy.random.randint(1, 10, size=m)
mu = 1.0 / (1.0 + numpy.exp(-eta))
y = numpy.random.binomial(n, mu)
model.fit(X, y, n=n)
assert scipy.stats.pearsonr(model.predict(X), eta) > .99
