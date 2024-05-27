import numpy as np

def sigma(y):
    return 1 / (1 + np.exp(-y))

def Bernoulli_sample(p):
    return np.random.binomial(1, p)

def pref_oracle(fx, fx_prime):
    p = sigma(fx - fx_prime)
    xwin = Bernoulli_sample(p)
    return xwin
