import numpy
from grm import GeneralizedRegressor, LinearRegressor,\
    NormalGrmConfiguration, BinomialGrmConfiguration
from pyearth import Earth
from sklearn.pipeline import Pipeline
from sklearn.linear_model.logistic import LogisticRegression


def test_linear_gaussian():
    m = 1000
    n = 10
    X = numpy.random.normal(size=(m,n))
    beta = numpy.random.normal(size = n)
    y = numpy.dot(X, beta) + 0.2*numpy.random.normal(size=m)
    model = GeneralizedRegressor(LinearRegressor(), NormalGrmConfiguration(), max_iter=10)
    model.fit(X, y)
    print beta
    print model.regressor_.coef_
    print model.record_


def test_linear_logistic():
    m = 10000
    n = 10
    X = numpy.random.normal(size=(m,n))
    beta = numpy.random.normal(size = n)**2
    eta = numpy.dot(X,beta)
    mu = 1.0 / (1.0 + numpy.exp(-eta))
    y = numpy.random.binomial(n=1,p=mu)
    model = GeneralizedRegressor(LinearRegressor(), BinomialGrmConfiguration(1), max_iter=10)
    model.fit(X, y)
    print beta
    print model.regressor_.coef_
    print model.record_
    
def test_earth_logistic():
    m = 10000
    n = 10
    X = numpy.random.normal(size=(m,n))
    eta = numpy.abs(X[:,6])
    mu = 1.0 / (1.0 + numpy.exp(-eta))
    y = numpy.random.binomial(n=1,p=mu)
    model = GeneralizedRegressor(Earth(penalty=10), BinomialGrmConfiguration(1), max_iter=10)
    model.fit(X, y)
    print model.regressor_.summary()
    print model.record_
    
if __name__ == '__main__':
    numpy.random.seed(1)
    test_linear_gaussian()
    test_linear_logistic()
    test_earth_logistic()
