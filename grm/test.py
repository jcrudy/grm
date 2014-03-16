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
    m = 1000
    n = 10
    X = 3*numpy.random.normal(size=(m,n))
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
    m = 1000
    n = 10
    X = 3*numpy.random.normal(size=(m,n))
    eta = 10*numpy.abs(X[:,6])
    mu = 1.0 / (1.0 + numpy.exp(-eta))
    y = numpy.random.binomial(n=1,p=mu)
    model = GeneralizedRegressor(Earth(), BinomialGrmConfiguration(1), max_iter=10)
    model.fit(X, y)
    print model.regressor_.summary()
    print model.record_
    
if __name__ == '__main__':
    numpy.random.seed(1)
    test_linear_gaussian()
    test_linear_logistic()
    test_earth_logistic()
    
    
    
#     X = 3*numpy.random.normal(size=(m,n))
# #     beta = numpy.random.normal(size = n)**2
#     eta = numpy.abs(X[:,6]) + 0.2*numpy.random.normal(size=m)
#     mu = 1.0 / (1.0 + numpy.exp(-eta))
#     y = numpy.random.binomial(n=1,p=mu)
#     model = GeneralizedRegressor(Earth(), BinomialGrmConfiguration(1), max_iter=10)
#     model.fit(X, y)
#     print model.record_
#     y_hat = model.predict(X)
#     model2 = Pipeline([('earth',Earth()), ('log',LogisticRegression())])
#     model2.fit(X,y)
#     y_hat2 = model2.predict_proba(X)[:,1]
#     print BinomialGrmConfiguration(1).score(y,y_hat), BinomialGrmConfiguration(1).score(y,y_hat2)
#     
#     
# #     
# #     print beta
# #     print model.regressor_.summary()
# #     print model.record_
# #     model2 = LogisticRegression()
# #     model2.fit(X,y)
# #     print model2.coef_
# #     model3 = GeneralizedRegressor(LinearRegressor(), BinomialGrmConfiguration(1), max_iter=10)
# #     X_ = X.copy()
# #     X_[:,6] = numpy.abs(X[:,6])
# #     model3.fit(X_,y)
# #     print model3.regressor_.coef_
# #     print model3.record_