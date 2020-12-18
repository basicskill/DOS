# TODO: Nadji mesto poklapanja circ i lin
# TODO: Nadji implementiranu circ conv
# TODO: Dodaj komentare

import numpy as np
from matplotlib import pyplot as plt

# Postavke zadatka
P = 3
N = 10 * (P + 1)

def lin_conv(x, y):
    N = len(x)

    x = np.concatenate((x, np.zeros(N)))
    y = np.concatenate((y, np.zeros(N)))

    lin = np.zeros(2 * N - 1)
    for n in range(2 * N - 1):
        lin[n] = sum([x[k] * y[n - k] for k in range(n + 1)])
    return lin

def circ_conv(x, y):
    circ = np.zeros(N)
    for n in range(N):
        circ[n] = sum([x[k] * y[(n - k) % N] for k in range(N)])
    return circ

if __name__ == "__main__":

    # Definicija x[n] signala
    x = np.arange(N)
    x[N//2:] = 1 - x[N//2:] - N/2

    # Definicija y[n] signala
    y = np.arange(N)
    y[:N//2] = 2 * np.cos((P + 1)*y[:N//2] + np.pi/4)
    y[N//2:] = 0

    # Plot ulaznih signala
    plt.stem(x, linefmt='gray', markerfmt='D')
    plt.stem(y, linefmt='gray', markerfmt='D')

    # Plot linearne konvolucije, implementirana naspram integrisana
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stem(lin_conv(x, y), linefmt='gray', markerfmt='D')
    plt.subplot(2, 1, 2)
    plt.stem(np.convolve(x, y), linefmt='gray', markerfmt='D')

    # Cirkularna konvolucija
    plt.figure()
    plt.stem(circ_conv(x, y), linefmt='gray', markerfmt='D')
    plt.show()