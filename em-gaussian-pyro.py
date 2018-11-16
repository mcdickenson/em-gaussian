# http://pyro.ai/examples/gmm.html

from collections import defaultdict
import numpy as np
import random as rand
import scipy.stats
import torch
from torch.distributions import constraints
from matplotlib import pyplot

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate


@config_enumerate(default='parallel')
@poutine.broadcast
def model(data):
    # Global variables.
    weights = pyro.param('weights', torch.FloatTensor([0.5]), constraint=constraints.unit_interval)
    scales = pyro.param('scales', torch.FloatTensor([2., 3., 4., 5.]), constraint=constraints.positive)
    locs = pyro.param('locs', torch.tensor([[1., 2.], [3., 4.]]))

    with pyro.iarange('data', data.size(0)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Bernoulli(torch.ones(len(data)) * weights))
        assignment = assignment.to(torch.int64)
        scales_assignment = scales[torch.stack((assignment*2, assignment*2 + 1))].transpose(1, 0)
        scales_assignment = torch.stack([torch.diag(s) for s in scales_assignment])
        locs_assignment = locs[assignment]
        obs = pyro.sample('obs', dist.MultivariateNormal(locs[assignment], scales_assignment), obs=data)


@config_enumerate(default="parallel")
@poutine.broadcast
def full_guide(data):
    with pyro.iarange('data', data.size(0)):
        # Local variables.
        assignment_probs = pyro.param('assignment_probs', torch.ones(len(data)) / K,
                                      constraint=constraints.unit_interval)
        pyro.sample('assignment', dist.Bernoulli(assignment_probs), infer={"enumerate": "sequential"})


def initialize(data):
    pyro.clear_param_store()

    optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    svi = SVI(model, full_guide, optim, loss=elbo)

    # Initialize weights to uniform.
    pyro.param('auto_weights', 0.5 * torch.ones(K), constraint=constraints.simplex)

    # Assume half of the data variance is due to intra-component noise.
    var = (data.var() / 2).sqrt()
    pyro.param('auto_scale', torch.tensor([var]*4), constraint=constraints.positive)

    # Initialize means from a subsample of data.
    pyro.param('auto_locs', data[torch.multinomial(torch.ones(len(data)) / len(data), K)]);

    loss = svi.loss(model, full_guide, data)

    return loss, svi


def get_samples():
    num_samples = 100

    # 2 clusters
    # note that both covariance matrices are diagonal
    mu1 = torch.tensor([0., 5.])
    sig1 = torch.tensor([ [2., 0.], [0., 3.] ])

    mu2 = torch.tensor([5., 0.])
    sig2 = torch.tensor([ [4., 0.], [0., 1.] ])

    # generate samples
    dist1 = dist.MultivariateNormal(mu1, sig1)
    samples1 = [pyro.sample('samples1', dist1) for _ in range(num_samples)]

    dist2 = dist.MultivariateNormal(mu2, sig2)
    samples2 = [pyro.sample('samples2', dist2) for _ in range(num_samples)]

    data = torch.cat((torch.stack(samples1), torch.stack(samples2)))
    return data


if __name__ == "__main__":
    pyro.enable_validation(True)
    pyro.set_rng_seed(42)

    # Create our model with a fixed number of components
    K = 2

    data = get_samples()

    global_guide = AutoDelta(poutine.block(model, expose=['weights', 'locs', 'scales']))
    global_guide = config_enumerate(global_guide, 'parallel')
    _, svi = initialize(data)

    for i in range(250):
        svi.step(data)

        if i % 150 == 0:
            print("locs: {}".format(pyro.param('locs')))
            print("scales: {}".format(pyro.param('scales')))
            print('weights = {}'.format(pyro.param('weights')))
            print('assignments: {}'.format(pyro.param('assignment_probs')))

    # todo plot data and estimates
