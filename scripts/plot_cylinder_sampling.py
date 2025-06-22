#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from fluxmd.core.cylindrical_sampler import FastCylindricalSampler

np.random.seed(0)

sampler = FastCylindricalSampler(100.0, 10.0)
pts = sampler.sample(5000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=2)
plt.show()
