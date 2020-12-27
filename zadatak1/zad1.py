import numpy as np
from matplotlib import pyplot as plt

# Postavke zadatka
P = 3
N = 10 * (P + 1)

def lin_conv(x, y):
    """ 
        Returns linear convolution of two descrete signals 
    """

    N = len(x)
    x = np.concatenate((x, np.zeros(N)))
    y = np.concatenate((y, np.zeros(N)))

    lin = np.zeros(2 * N - 1)
    for n in range(2 * N - 1):
        lin[n] = sum([x[k] * y[n - k] for k in range(n + 1)])
    return lin

def circ_conv(x, y):
    """ 
        Returns circular convolution of two descrete signals 
    """

    N = len(x)
    circ = np.zeros(N)
    for n in range(N):
        circ[n] = sum([x[k] * y[(n - k) % N] for k in range(N)])
    return circ

def cconv(x, y):
    """ Imitation of MATLAB's cconv function """
    return np.real(np.fft.ifft( np.fft.fft(x)*np.fft.fft(y) ))

if __name__ == "__main__":

    # Definicija x[n] signala
    x = np.arange(N)
    x[N//2:] = 1 - x[N//2:] - N/2

    # Definicija y[n] signala
    y = np.arange(N, dtype=float)
    y[:N//2] = 2.0 * np.cos((P + 1)*y[:N//2] + np.pi/4)
    y[N//2:] = 0

    # Plot ulaznih signala
    plt.stem(x, linefmt='gray', markerfmt='D', label='x[n]')
    plt.stem(y, linefmt='gray', markerfmt='D', label='y[n]')
    plt.xlabel('n')
    plt.title('Signali x[n] i y[n]')
    plt.legend()
    plt.savefig('figures/zad1_signali.png')

    # Linearna konvolucija
    lin = lin_conv(x, y)

    # Plot linearne konvolucije
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Linearna konvolucija')
    plt.stem(lin, linefmt='gray', markerfmt='D')
    plt.xlabel('n')
    plt.ylabel('Implementirana funkcija')

    plt.subplot(2, 1, 2)
    plt.stem(np.convolve(x, y), linefmt='gray', markerfmt='D')
    plt.xlabel('n')
    plt.ylabel('conv(x, y)')
    plt.savefig('figures/zad1_linearna_konvolucija.png')

    # Ciklicna konvoluciju
    circ = circ_conv(x, y)

    # Ciklicna konvolucija
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Ciklicna konvolucija')
    plt.stem(circ, linefmt='gray', markerfmt='D')
    plt.xlabel('n')
    plt.ylabel('Implementirana funkcija')

    plt.subplot(2, 1, 2)
    plt.stem(cconv(x, y), linefmt='gray', markerfmt='D')
    plt.xlabel('n')
    plt.ylabel('cconv(x, y)')
    plt.savefig('figures/zad1_ciklicna_konvolucija_provera.png')

    # Overlap dve funkcije
    overlap = lin * np.isin(lin, circ)
    overlap[overlap == 0] = None

    plt.figure()
    plt.title('Preklop linearne i ciklicne konvolucije')
    plt.stem(lin, linefmt='gray', markerfmt='D', label='Linearna kovolucija')
    plt.stem(overlap, linefmt='blue', markerfmt='rD', label='Preklop')
    plt.xlabel('n')
    plt.legend()
    plt.savefig('figures/zad1_overlap.png')