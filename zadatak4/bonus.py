import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io.wavfile import read, write


def estimate_pitch(x, fs):
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
    spek /= max(spek)
    freq = (np.arange(len(spek)) * fs / step).flatten()

    lowbound   = np.nonzero(freq < 165)[-1][-1]
    highbound  = np.nonzero(freq < 255)[-1][-1]

    return freq[lowbound + np.argmax(spek[lowbound:highbound+1])]



if __name__ == "__main__":
    # Ucitavanje originalnog signala
    fs, x = read('sounds/in.wav')

    print(f'Pitch frekvencija je {estimate_pitch(x, fs)}')