import numpy
from grm import GeneralizedRegressor, MidpointLogHazardLossFunction
from samplers import HazardSampler
from matplotlib import pyplot
import scipy.stats
from pyearth.earth import Earth
import pickle
import os
numpy.seterr(all='raise')
numpy.random.seed(1)

data_filename = 'log_hazard_data.pickle'
redo = False
if os.path.exists(data_filename) and not redo:
    with open(data_filename, 'r') as infile:
        m, y, c, censor_times, failure_times = pickle.load(infile)
else:
    m = 100000
    censor_times = numpy.random.uniform(0.0, 100.0, size=m)
    baseline_hazard = lambda t: numpy.exp(numpy.sin(t) - 2.0)
    sampler = HazardSampler(baseline_hazard, 10.0, 20.0)
    failure_times = numpy.array([sampler.draw() for _ in range(m)])
    y = numpy.minimum(failure_times, censor_times)
    c = 1.0 * (censor_times > failure_times)
    with open(data_filename, 'w') as outfile:
        pickle.dump((m, y, c, censor_times, failure_times), outfile)
pyplot.hist(failure_times, bins=50)
pyplot.show()
model = GeneralizedRegressor(base_regressor=Earth(penalty=0.0, allow_linear=False, max_degree=2), loss_function=MidpointLogHazardLossFunction())
model.fit(X=None,y=y,c=c)
t = numpy.arange(0.0, 20.0, .1)
predicted_log_hazard = model.predict(X=None, t=t)
actual_log_hazard = numpy.sin(t) - 2.0
pyplot.figure()
pyplot.plot(t, actual_log_hazard, 'r', label='actual log hazard')
pyplot.plot(t, predicted_log_hazard, 'b', label='predicted log hazard')
pyplot.show()
print model.regressor_.trace()
print model.regressor_.summary()


