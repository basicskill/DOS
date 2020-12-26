import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read

# Postavka
S = 1 # Slovo E

def segment_fft(x, fs):
    """ Returns fft of 20ms window """

    step = int(fs * 20 * 1e-3)

    spek = []

    for idx in range(0, step * (len(x) // step), step):
        X = np.fft.fft(x[idx:idx + step])
        X = X[:len(X) // 2] / max(X)
        X = abs(X)

        if len(spek) == 0:
            spek = X
        else:
            spek += X

    spek /= len(x) / step
    freq = (np.arange(len(spek)) * fs / step).flatten()

    return freq, spek
    


if __name__ == "__main__":
    # Ucitavanje originalnog signala
    fs, x = read('in.wav')
    time = np.arange(0, len(x) / fs, 1 / fs)
    time = time[:-1]

    freq, spek = segment_fft(x, fs)
    plt.plot(freq[:len(freq) // 2], spek[:len(spek) // 2])
    plt.show()