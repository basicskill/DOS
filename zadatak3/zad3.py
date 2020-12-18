# TODO: Analog -> Digital filter? Odma iz funkcije ili raditi neku zavrzlamu?
import scipy
from scipy.io import loadmat
from matplotlib import pyplot as plt

P = 3
R = 1

data = loadmat('ekg' + str(P) + '.mat')
x = data['x']
fs = data['fs']


plt.plot(x)
plt.show()