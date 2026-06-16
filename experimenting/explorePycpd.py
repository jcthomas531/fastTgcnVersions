from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration
import numpy as np

import os
os.chdir("H:/schoolFiles/clonedRepos/pycpd")


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


fish_target = np.loadtxt('data/fish_target.txt')
X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
X1[:, :-1] = fish_target
X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
X2[:, :-1] = fish_target
X = np.vstack((X1, X2))

fish_source = np.loadtxt('data/fish_source.txt')
Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
Y1[:, :-1] = fish_source
Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
Y2[:, :-1] = fish_source
Y = np.vstack((Y1, Y2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
callback = partial(visualize, ax=ax)

reg = DeformableRegistration(**{'X': X, 'Y': Y})
reg.register(callback)
plt.show()





aaa = reg.register(Y)[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c="red")
ax.scatter(Y[:,0], Y[:,1], Y[:,2], c="blue")
ax.scatter(aaa[:,0], aaa[:,1], aaa[:,2], c="green")
plt.show()


from pycpd import RigidRegistration
regRigid = RigidRegistration(**{'X': X, 'Y': Y})

bbb = regRigid.register(Y)[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c="red")
ax.scatter(Y[:,0], Y[:,1], Y[:,2], c="blue")
ax.scatter(bbb[:,0], bbb[:,1], bbb[:,2], c="green")
plt.show()


from pycpd import AffineRegistration
regAffine = AffineRegistration(**{'X': X, 'Y': Y})

ccc = regRigid.register(Y)[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c="red")
ax.scatter(Y[:,0], Y[:,1], Y[:,2], c="blue")
ax.scatter(ccc[:,0], ccc[:,1], ccc[:,2], c="green")
plt.show()


dir(reg)
reg.alpha
reg.beta
reg.diff
reg.expectation()
reg.get_registration_parameters()
reg.iterate()
reg.max_iterations
reg.maximization()
reg.q
reg.register()
reg.sigma2
reg.tolerance
reg.transform_point_cloud()
reg.update_transform()
reg.update_variance()
reg.w
