import numpy
import scipy.integrate

class HazardSampler(object):
    def __init__(self, hazard, start=0.0, step=None):
        self.hazard = hazard
        if step is None:
            h0 = hazard(0.0)
            if h0 > 0:
                step = 2.0 / hazard(0.0)
            else:
                # Reasonable default.  Not efficient in some cases.
                step = 200.0 / scipy.integrate.quad(hazard, 0.0, 100.0) 
        self.cumulative_hazard = CumulativeHazard(hazard)
        self.survival_function = SurvivalFunction(self.cumulative_hazard)
        self.cdf = Cdf(self.survival_function)
        self.inverse_cdf = InverseCdf(self.cdf, start=start, step=step, lower=0.0)
        self.sampler = InversionTransformSampler(self.inverse_cdf)

    def draw(self):
        return self.sampler.draw()

class InversionTransformSampler(object):
    def __init__(self, inverse_cdf):
        self.inverse_cdf = inverse_cdf

    def draw(self):
        u = numpy.random.uniform(0,1)
        return self.inverse_cdf(u)

class CumulativeHazard(object):
    def __init__(self, hazard):
        self.hazard = hazard

    def __call__(self, t):
        return scipy.integrate.quad(self.hazard, 0.0, t)[0]

class SurvivalFunction(object):
    def __init__(self, cumulative_hazard):
        self.cumulative_hazard = cumulative_hazard

    def __call__(self, t):
        return numpy.exp(-self.cumulative_hazard(t))

class Cdf(object):
    def __init__(self, survival_function):
        self.survival_function = survival_function

    def __call__(self, t):
        return 1.0 - self.survival_function(t)

class InverseCdf(object):
    def __init__(self, cdf, start, step, precision=1e-8, lower=float('-inf'),
                 upper=float('inf')):
        self.cdf = cdf
        self.precision = precision
        self.start = start
        self.step = step
        self.lower = lower
        self.upper = upper

    def __call__(self, p):
        last_diff = None
        step = self.step
        current = self.start
        while True:
            value = self.cdf(current)
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


        