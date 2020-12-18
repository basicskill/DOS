import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
from scipy.fft import *

Q = 3
fs = 4400

# Ucitavanje originalnog signala
fs_original, x = read('audio' + str(Q) + '.wav')
time_original = np.arange(0, len(x) / fs_original, 1 / fs_original)

# Odabiranje signala zadatom frekvencijom
x_sampled = x[::fs_original // fs]
time_sampled  = np.arange(0, len(x_sampled) / fs, 1 / fs)

# Prikaz vremenskih oblika odabranog i ne odabranog signala
# plt.subplot(2, 1, 1)
# plt.plot(time_original, x)
# plt.subplot(2, 1, 2)
# plt.plot(time_sampled, x_sampled)
# plt.show()

X = fftshift(fft(x))
freq = fftshift(fftfreq(time_original.shape[-1]))
plt.plot(freq[len(freq)//2:], abs(X.real[len(X) // 2:]))
plt.show()