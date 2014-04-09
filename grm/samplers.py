import numpy
import scipy.integrate


class HazardSampler(object):
    def __init__(self, hazard, mu, sigma):
        self.hazard = hazard
        self.cumulative_hazard = CumulativeHazard(hazard)
        self.survival_function = SurvivalFunction(self.cumulative_hazard)
        self.cdf = Cdf(self.survival_function)
        self.inverse_cdf = InverseCdf(self.cdf, mu=mu, sigma=sigma, lower=0.0)
        self.sampler = InversionTransformSampler(self.inverse_cdf)

    def draw(self, **kwargs):
        return self.sampler.draw()

class InversionTransformSampler(object):
    def __init__(self, inverse_cdf):
        self.inverse_cdf = inverse_cdf

    def draw(self, **kwargs):
        u = numpy.random.uniform(0,1)
        return self.inverse_cdf(u, **kwargs)

class CumulativeHazard(object):
    def __init__(self, hazard):
        self.hazard = hazard

    def __call__(self, t, **kwargs):
        if kwargs:
            func = functools.partial(self.hazard, **kwargs)
        else:
            func = self.hazard
        return scipy.integrate.quad(func, 0.0, t)[0]

class SurvivalFunction(object):
    def __init__(self, cumulative_hazard):
        self.cumulative_hazard = cumulative_hazard

    def __call__(self, t, **kwargs):
        return numpy.exp(-self.cumulative_hazard(t, **kwargs))

class Cdf(object):
    def __init__(self, survival_function):
        self.survival_function = survival_function

    def __call__(self, t, **kwargs):
        return 1.0 - self.survival_function(t, **kwargs)

class InverseCdf(object):
    def __init__(self, cdf, mu, sigma, precision=1e-8, lower=float('-inf'),
                 upper=float('inf')):
        self.cdf = cdf
        self.precision = precision * sigma
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper

    def __call__(self, p, **kwargs):
        last_diff = None
        step = self.sigma
        current = self.mu
        while True:
            value = self.cdf(current, **kwargs)
            diff = value - p
            if abs(diff) < self.precision:
                break
            elif diff < 0:
                current = min(current + step, self.upper)
                if last_diff is not None and last_diff > 0:
                    step *= 0.5
                last_diff = diff
            else:
                current = max(current - step, self.lower)
                if last_diff is not None and last_diff < 0:
                    step *= 0.5
                last_diff = diff
        return current




# class RejectionSampler(object):
#     def __init__(self, target, proposal, factor=None, lower=-10000.0, upper=10000.0, n_points=100000):
#         self.target = target
#         self.proposal = proposal
#         if factor is not None:
#             self.factor = factor
#         else:
#             self.factor = self.find_factor(lower, upper, n_points)
#
#     def find_factor(self, lower, upper, n_points):
#         points = numpy.linspace(lower, upper, n_points)
#         ratios = proposal.pdf(points) / target(points)
#         return numpy.max(ratios)
#
#     def rvs(self, size=1):
#         result = numpy.empty(shape=size,dtype=float)
#         x = proposal.rvs(size=size)
#         reject = numpy.ones(shape=size).astype(bool)
#         unifs = numpy.random.uniform(size=size)
#         while True:
#             ratios[reject] = proposal.pdf(x[reject]) * self.factor / target(x[reject])
#             unifs = numpy.random.uniform(size=size)[reject]
#             reject = ratios > unifs
#             if not numpy.any(reject):
#                 break
#         return x
#
#
# class SurvivalDensity(object):
#     def __init__(self, hazard, cum_hazard):
#         self.hazard = hazard
#         self.cum_hazard = cum_hazard
#
#     def __call__(self, t):
#         return self.hazard(t) * numpy.exp(-self.cum_hazard(t))
#
# class HazardSampler(object):
#     def __init__(self, hazard, cum_hazard=None, lower=0.0, upper=10000.0, n_points=100000):
#         self.hazard = hazard
#         self.cum_hazard = cum_ha
#         self.lower = lower
#         self.upper = upper
#         self.n_points = n_points
#         self.density = SurvivalDensity()
#
#     def density(self, t, **kwargs):
#
#
#
#     def rvs()
