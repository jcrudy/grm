import numpy
from grm import GeneralizedRegressor, MidpointLogHazardLossFunction
from samplers import HazardSampler
from matplotlib import pyplot
import scipy.stats
from pyearth.earth import Earth
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
import time
numpy.seterr(all='raise')
numpy.random.seed(1)


m = 1000
data_filename = 'log_hazard_data' + str(m) + '.pickle'
redo = False
if os.path.exists(data_filename) and not redo:
    with open(data_filename, 'r') as infile:
        m, y, c, censor_times, failure_times = pickle.load(infile)
else:
    censor_times = numpy.random.uniform(0.0, 100.0, size=m)
    baseline_hazard = lambda t: numpy.exp(numpy.sin(t) - 2.0)
    sampler = HazardSampler(baseline_hazard, 10.0, 20.0)
    failure_times = numpy.array([sampler.draw() for _ in range(m)])
    y = numpy.minimum(failure_times, censor_times)
    c = 1.0 * (censor_times > failure_times)
    with open(data_filename, 'w') as outfile:
        pickle.dump((m, y, c, censor_times, failure_times), outfile)
pyplot.hist(y, bins=50)
pyplot.show()
t0 = time.time()
model = GeneralizedRegressor(base_regressor=Earth(thresh=1e-7, max_terms=100, smooth=True, allow_linear=False, penalty=0), loss_function=MidpointLogHazardLossFunction(10))
model.fit(X=None,y=y,c=c)
t1 = time.time()
print 'Total fitting time: %f seconds' % (t1 - t0)
t = numpy.arange(0.0, 30.0, .1)
predicted_log_hazard = model.predict(X=None, t=t)
actual_log_hazard = numpy.sin(t) - 2.0
pyplot.figure()
pyplot.plot(t, actual_log_hazard, 'r', label='actual log hazard')
pyplot.plot(t, predicted_log_hazard, 'b', label='predicted log hazard')
pyplot.show()
try:
    print model.regressor_.trace()
    print model.regressor_.summary()
except:
    pass

