import numpy
from grm import GeneralizedRegressor, BinomialLossFunction, LogitLink
import scipy.stats
from pyearth.earth import Earth


numpy.random.seed(1)
m = 1000
n = 10
p = 10
t = numpy.random.uniform(0.0,1000.0,size=m)
x1 = numpy.random.uniform(0.0,1.0,size=m)
x2 = numpy.random.uniform(0.0,1.0,size=m)
baseline_log_hazard = numpy.sin(t) + numpy.abs(x1)
X = 
