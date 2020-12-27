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

    # Spektar
    X = np.fft.fft(x)
    X = X[range(len(X) // 8)] / max(X)
    freq = (np.arange(len(X)) * fs / len(x)).flatten()

    ## Filtriranje

    # Elliptic filter
    be, ae = signal.ellip(4, 2, 40, [45, 55], btype='bandstop', analog=True)
    bze, aze = signal.bilinear(be, ae, fs / (2 * np.pi))
    
    # Filtriranje
    x_ellip = signal.filtfilt(bze, aze, x)

    # Racunanje spektra
    X_ellip = np.fft.fft(x_ellip)
    X_ellip = X_ellip[range(len(X_ellip) // 8)] / max(X_ellip)
    freq = (np.arange(len(X_ellip)) * fs / len(x_ellip)).flatten()

    # Cheby I filter
    bc, ac = signal.cheby1(4, 2, [45, 55], btype='bandstop', analog=True)
    bzc, azc = signal.bilinear(bc, ac, fs / (2 * np.pi))

    # Filtriranje
    x_cheby1 = signal.filtfilt(bzc, azc, x)

    # Racunanje spektra
    X_cheby1 = np.fft.fft(x_cheby1)
    X_cheby1 = X_cheby1[range(len(X_cheby1) // 8)] / max(X_cheby1)

    ## Racunanje broja otkucaja u minuti

    # Tresholdovanje signala
    trashhold = 0.8 * max(x_cheby1)
    thr = x_cheby1 > trashhold

    # Brojanje pikova
    cnt = sum([1 for idx, _ in enumerate(thr[1:]) if thr[idx - 1] and not thr[idx]])

    # Rezultat
    print(f'Broj otkucaja u minuti je {int(cnt * (60 / t[-1]))}')


    ## Plotovanje razultata

    # a) signal vreme
    plt.xlabel('vreme [s]')
    plt.ylabel('EKG')
    plt.title('Nefiltrirani signal')
    plt.plot(t, x)

    plt.savefig('figures/zad3_signal_vreme.png')

    # b) spektar signal
    plt.figure()
    plt.stem(freq, abs(X), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Originalni spektar')

    plt.savefig('figures/zad3_signal_spektar.png')

    # c) AFK filtera
    plt.figure()

    plt.subplot(2, 1, 1)
    w, h = signal.freqs(be, ae)
    wz, hz = signal.freqz(bze, aze)
    plt.semilogx(w, 20 * np.log10(abs(h)), label='analogni')
    plt.semilogx((wz * fs / (2 * np.pi)).T, 20 * np.log10(abs(hz)), label='digitalni')
    plt.legend()
    plt.title('Elipticni bandstop filteri')
    plt.xlabel('Frekvencija [rad / sec]')
    plt.ylabel('Amplituda [dB]')

    plt.subplot(2, 1, 2)
    w, h = signal.freqs(bc, ac)
    wz, hz = signal.freqz(bzc, azc)
    plt.semilogx(w, 20 * np.log10(abs(h)), label='analog')
    plt.semilogx((wz * fs / (2 * np.pi)).T, 20 * np.log10(abs(hz)), label='digital')
    plt.legend()
    plt.title('Chebisev I bandstop filteri')
    plt.xlabel('Frekvencija [rad / sec]')
    plt.ylabel('Amplituda [dB]')

    plt.tight_layout()
    plt.savefig('figures/zad3_filteri.png')

    # d) svi signali u vremenu
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.xlabel('vreme [s]')
    plt.ylabel('EKG')
    plt.title('Nefiltrirani signal')
    plt.plot(t, x)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, x_ellip)
    plt.xlabel('vreme [s]')
    plt.ylabel('EKG')
    plt.title('Signal na izlazu Elipticnog filtera')

    plt.subplot(3, 1, 3)
    plt.plot(t, x_cheby1)
    plt.xlabel('vreme [s]')
    plt.ylabel('EKG')
    plt.title('Signal na izlazu Cheby I filtera')

    plt.tight_layout()
    plt.savefig('figures/zad3_filtrirani_vreme.png')

    # e) AFK svih signala
    plt.figure()
    plt.tight_layout()

    plt.subplot(3, 1, 1)
    plt.stem(freq, abs(X), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Originalni spektar')

    plt.subplot(3, 1, 2)
    plt.stem(freq, abs(X_ellip), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Spektar na izlazu Elipticnog filtera')
    
    plt.subplot(3, 1, 3)
    plt.stem(freq, abs(X_cheby1), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Spektar na izlazu Cheby I filtera')

    plt.tight_layout()
    plt.savefig('figures/zad3_filtrirani_spektar.png')