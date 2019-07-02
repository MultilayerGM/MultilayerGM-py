import random as _rand


def categorical(cum_weights):
    """
    Sample from a discrete distribution

    :param cum_weights: list of cumulative sums of probabilities (should satisfy cum_weights[-1] = 1)
    :return: sampled integer from `range(0,len(cum_weights))`
    """
    p = _rand.random()
    i = 0
    while p > cum_weights[i]:
        i += 1
    return i


def dirichlet(theta, n):
    """
    Sample from a symmetric dirichlet distribution

    :param theta: concentration parameter (theta > 0)
    :param n: number of variables
    :return: list of sampled probabilities of length n
    """
    weights = [_rand.gammavariate(theta,1) for _ in range(n)]
    s = sum(weights)
    weights[:] = [w/s for w in weights]
    return weights
