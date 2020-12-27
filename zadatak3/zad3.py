import numpy as np
from scipy import signal
from scipy.io import loadmat
from matplotlib import pyplot as plt

P = 3
R = 1 # Eliptic i Cheb I


if __name__ == "__main__":

    ## Ucitavanje podataka

    # Vremenski signal
    data = loadmat('input/ekg' + str(R) + '.mat')
    x    = data['x'].flatten()
    fs   = 2 * data['fs']
    t    = (np.arange(len(x)) / fs).flatten()

    # Plot originalnog signala
    plt.xlabel('vreme [s]')
    plt.ylabel('EKG')
    plt.title('Nefiltrirani signal')
    plt.plot(t, x)
    plt.savefig('figures/zad3_signal_vreme.png')

    # Spektar
    X = np.fft.fft(x)
    X = X[range(len(X) // 8)] / max(X)
    freq = (np.arange(len(X)) * fs / len(x)).flatten()

    plt.figure()
    plt.stem(freq, abs(X), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Originalni spektar')
    plt.savefig('figures/zad3_signal_spektar.png')

    ## Filtriranje

    # Elliptic filter
    b, a = signal.ellip(4, 2, 40, [45, 55], btype='bandstop', analog=True)
    b1, a1 = signal.bilinear(b, a, fs / (2 * np.pi))

    # Plot Eliptic filtera
    plt.figure()
    w, h = signal.freqs(b, a)
    wz, hz = signal.freqz(b1, a1)
    plt.semilogx(w, 20 * np.log10(abs(h)), label='analogni')
    plt.semilogx((wz * fs / (2 * np.pi)).T, 20 * np.log10(abs(hz)), label='digitalni')
    plt.legend()
    plt.title('Elipticni bandstop filteri')
    plt.xlabel('Frekvencija [rad / sec]')
    plt.ylabel('Amplituda [dB]')
    plt.savefig('figures/zad3_ellip_filteri.png')
    
    # Filtriranje
    x_ellip = signal.filtfilt(b1, a1, x)

    plt.figure()
    plt.plot(t, x_ellip)
    plt.xlabel('vreme [s]')
    plt.ylabel('EKG')
    plt.title('Signal na izlazu Elipticnog filtera')
    plt.savefig('figures/zad3_ellip_vreme.png')

    # Racunanje spektra
    X_ellip = np.fft.fft(x_ellip)
    X_ellip = X_ellip[range(len(X_ellip) // 8)] / max(X_ellip)
    freq = (np.arange(len(X_ellip)) * fs / len(x_ellip)).flatten()

    plt.figure()
    plt.stem(freq, abs(X_ellip), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Originalni spektar')
    plt.title('Spektar na izlazu Elipticnog filtera')
    plt.savefig('figures/zad3_ellip_spektar.png')


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
    plt.title('Chebisev I bandstop filteri')
    plt.xlabel('Frekvencija [rad / sec]')
    plt.ylabel('Amplituda [dB]')
    plt.savefig('figures/zad3_cheby1_filteri.png')
    
    # Filtriranje
    x_cheby1 = signal.filtfilt(b1, a1, x)

    # Plotovanje signala
    plt.figure()
    plt.plot(t, x_cheby1)
    plt.xlabel('vreme [s]')
    plt.ylabel('EKG')
    plt.title('Signal na izlazu Cheby I filtera')
    plt.savefig('figures/zad3_cheby1_vreme.png')

    # Racunanje spektra
    X_cheby1 = np.fft.fft(x_cheby1)
    X_cheby1 = X_ellip[range(len(X_cheby1) // 8)] / max(X_cheby1)
    freq = (np.arange(len(X_cheby1)) * fs / len(x_cheby1)).flatten()

    # Plotovanje spektra
    plt.figure()
    plt.stem(freq, abs(X_cheby1), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Spektar na izlazu Cheby I filtera')
    plt.savefig('figures/zad3_cheby1_spektar.png')

    ## Racunanje broja otkucaja u minuti

    # Tresholdovanje signala
    trashhold = 0.8 * max(x_cheby1)
    thr = x_cheby1 > trashhold

    # Brojanje pikova
    cnt = sum([1 for idx, _ in enumerate(thr[1:]) if thr[idx - 1] and not thr[idx]])

    # Rezultat
    print(f'Broj otkucaja u minuti je {int(cnt * (60 / t[-1]))}')