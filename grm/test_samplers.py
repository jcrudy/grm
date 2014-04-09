from .samplers import InverseCdf, HazardSampler
from nose.tools import assert_almost_equal
import scipy.stats
import numpy
from matplotlib import pyplot
from statsmodels.distributions import ECDF

numpy.random.seed(1)

class TestInverseCdf(object):
    test_size = 100
    def test_normal_cdf(self):
        cdf = scipy.stats.norm.cdf
        q = scipy.stats.norm.ppf
        inverter = InverseCdf(cdf, 0.0, 1.0)
        for _ in range(self.test_size):
            u = numpy.random.uniform()
            q_ = inverter(u)
            assert_almost_equal(q(u),  q_, 5)


class TestHazardSampler(object):
    test_size = 100
    def test_constant_hazard(self):
        '''
        Test against a constant hazard function, which should give an
        exponential distribution.
        '''
        hazard = lambda t: 1.0
        sampler = HazardSampler(hazard, 1.0, 1.0)

        for _ in range(self.test_size):
            u = numpy.random.uniform()
            t = scipy.stats.expon.ppf(u)
            s = 1.0 - u
            assert_almost_equal(sampler.cumulative_hazard(t), t, 5)
            assert_almost_equal(sampler.survival_function(t), s, 5)
            assert_almost_equal(sampler.cdf(t), u, 5)
            assert_almost_equal(sampler.inverse_cdf(u), t, 5)

        sample = []
        for __ in range(10000):
            sample.append(sampler.draw())
        sample = numpy.array(sample)
        cdf = ECDF(sample)
        points = numpy.arange(0.0, 10.0, .1)
        points_est = cdf(points)
        points_expon = scipy.stats.expon.cdf(points)
        numpy.testing.assert_almost_equal(points_est, points_expon, 2)
