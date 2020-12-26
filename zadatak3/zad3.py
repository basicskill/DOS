import numpy as np
from scipy import signal
from scipy.io import loadmat
from matplotlib import pyplot as plt

P = 3
R = 1 # Eliptic i Cheb I


if __name__ == "__main__":

    ## Ucitavanje podataka

    # Vremenski signal
    data = loadmat('ekg' + str(R) + '.mat')
    x    = data['x'].flatten()
    fs   = 2 * data['fs']
    t    = (np.arange(len(x)) / fs).flatten()


    plt.plot(t, x)
    plt.savefig('figures/zad2_signal_vreme.png')


    # Spektar
    X = np.fft.fft(x)
    X = X[range(len(X) // 8)] / max(X)
    freq = (np.arange(len(X)) * fs / len(x)).flatten()


    plt.figure()
    plt.plot(freq, abs(X))
    plt.title('Originalni spektar')
    plt.savefig('figures/zad2_signal_spektar.png')

    ## Filtriranje

    # Elliptic filter
    b, a = signal.ellip(4, 2, 40, [45, 55], btype='bandstop', analog=True)
    b1, a1 = signal.bilinear(b, a, fs / (2 * np.pi))

    # Plot analog filter
    plt.figure()
    w, h = signal.freqs(b, a)
    wz, hz = signal.freqz(b1, a1)
    plt.semilogx(w, 20 * np.log10(abs(h)), label='analog')
    plt.semilogx((wz * fs / (2 * np.pi)).T, 20 * np.log10(abs(hz)), label='digital')
    plt.legend()
    plt.title('Filteri')
    plt.savefig('figures/zad2_ellip_filteri.png')
    
    x_ellip = signal.filtfilt(b1, a1, x)

    plt.figure()
    plt.plot(t, x_ellip)
    plt.savefig('figures/zad2_ellip_vreme.png')

    X_ellip = np.fft.fft(x_ellip)
    X_ellip = X_ellip[range(len(X_ellip) // 8)] / max(X_ellip)
    freq = (np.arange(len(X_ellip)) * fs / len(x_ellip)).flatten()

    plt.figure()
    plt.plot(freq, abs(X_ellip))
    plt.title('Filtrirani spektar ellip')
    plt.savefig('figures/zad2_ellip_spektar.png')


    # Cheby I filter
    b, a = signal.cheby1(4, 2, [45, 55], btype='bandstop', analog=True)
    b1, a1 = signal.bilinear(b, a, fs / (2 * np.pi))

    # Plot analog filter
    plt.figure()
    w, h = signal.freqs(b, a)
    wz, hz = signal.freqz(b1, a1)
    plt.semilogx(w, 20 * np.log10(abs(h)), label='analog')
    plt.semilogx((wz * fs / (2 * np.pi)).T, 20 * np.log10(abs(hz)), label='digital')
    plt.legend()
    plt.title('Filteri')
    plt.savefig('figures/zad2_cheby1_filteri.png')
    
    x_cheby1 = signal.filtfilt(b1, a1, x)

    plt.figure()
    plt.plot(t, x_cheby1)
    plt.savefig('figures/zad2_cheby1_vreme.png')

    X_cheby1 = np.fft.fft(x_cheby1)
    X_cheby1 = X_ellip[range(len(X_cheby1) // 8)] / max(X_cheby1)
    freq = (np.arange(len(X_cheby1)) * fs / len(x_cheby1)).flatten()

    plt.figure()
    plt.plot(freq, abs(X_ellip))
    plt.title('Filtrirani spektar ellip')
    plt.savefig('figures/zad2_cheby1_spektar.png')

    ## Brojanje pikova
    trashhold = 0.8 * max(x_cheby1)
    thr = x_cheby1 > trashhold

    plt.close('all')
    plt.plot(x_cheby1)
    plt.axhline(y=trashhold, color='r')
    plt.show()
    
    cnt = sum([1 for idx, _ in enumerate(thr[1:]) if thr[idx - 1] and not thr[idx]])

    print(cnt)