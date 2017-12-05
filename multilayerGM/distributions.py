import random as _rand


def categorical(cum_weights):
    p = _rand.random()
    i = 0
    while p > cum_weights[i]:
        i += 1
    return i


def dirichlet(theta, n):
    weights = [_rand.gammavariate(theta,1) for _ in range(n)]
    s = sum(weights)
    weights[:] = [w/s for w in weights]
    return weights
