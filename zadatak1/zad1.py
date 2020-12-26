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
    N = len(x)
    circ = np.zeros(N)
    for n in range(N):
        circ[n] = sum([x[k] * y[(n - k) % N] for k in range(N)])
    return circ

if __name__ == "__main__":

    # Definicija x[n] signala
    x = np.arange(N)
    x[N//2:] = 1 - x[N//2:] - N/2

    # Definicija y[n] signala
    y = np.arange(N, dtype=float)
    y[:N//2] = 2.0 * np.cos((P + 1)*y[:N//2] + np.pi/4)
    y[N//2:] = 0

    # Plot ulaznih signala
    plt.stem(x, linefmt='gray', markerfmt='D')
    plt.stem(y, linefmt='gray', markerfmt='D')
    plt.savefig('figures/zad1_signali.png')

    # Linearna konvolucija
    lin = lin_conv(x, y)

    # Plot linearne konvolucije, implementirana naspram integrisana
    plt.figure()
    plt.stem(lin, linefmt='gray', markerfmt='D')
    plt.savefig('figures/zad1_linearna_konvolucija.png')

    # Cirkularnu konvoluciju
    circ = circ_conv(x, y)

    # Cirkularna konvolucija
    plt.figure()
    plt.stem(circ, linefmt='gray', markerfmt='D')
    plt.savefig('figures/zad1_ciklicna_konvolucija.png')


    # overlap = np.isin(lin, circ)
    # plt.figure()
    # plt.stem(circ, linefmt='gray', markerfmt='D')
    # plt.stem(lin * overlap, linefmt='gray', markerfmt='D')