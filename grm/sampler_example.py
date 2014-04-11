# Produce some simulated survival data from a weird hazard function
import numpy
from samplers import HazardSampler

# Set a random seed and sample size
numpy.random.seed(1)
m = 1000

# Use this totally crazy hazard function
hazard = lambda t: numpy.exp(numpy.sin(t) - 2.0)

# Sample failure times from the hazard function
sampler = HazardSampler(hazard)
failure_times = numpy.array([sampler.draw() for _ in range(m)])

# Apply some non-informative right censoring, just to demonstrate how it's done
censor_times = numpy.random.uniform(0.0, 25.0, size=m)
y = numpy.minimum(failure_times, censor_times)
c = 1.0 * (censor_times > failure_times)

# Make some plots of the simulated data
from matplotlib import pyplot
from statsmodels.distributions import ECDF

# Plot a histogram of failure times from this hazard function
pyplot.hist(failure_times, bins=50)
pyplot.title('Uncensored Failure Times')
pyplot.savefig('uncensored_hist.png', transparent=True)
pyplot.show()

# Plot a histogram of censored failure times from this hazard function
pyplot.hist(y, bins=50)
pyplot.title('Non-informatively Right Censored Failure Times')
pyplot.savefig('censored_hist.png', transparent=True)
pyplot.show()

# Plot the empirical survival function (based on the censored sample) against the actual survival function
t = numpy.arange(0,20.0,.1)
S = numpy.array([sampler.survival_function(t[i]) for i in range(len(t))])
S_hat = 1.0 - ECDF(failure_times)(t)
pyplot.figure()
pyplot.title('Survival Function Comparison')
pyplot.plot(t, S, 'r', lw=3, label='True survival function')
pyplot.plot(t, S_hat, 'b--', lw=3, label='Sampled survival function (1 - ECDF)')
pyplot.legend()
pyplot.xlabel('Time')
pyplot.ylabel('Proportion Still Alive')
pyplot.savefig('survival_comp.png', transparent=True)
pyplot.show()




