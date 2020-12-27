import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io.wavfile import read, write

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
    spek /= max(spek)
    freq = (np.arange(len(spek)) * fs / step).flatten()

    return freq, spek

if __name__ == "__main__":
    # Ucitavanje originalnog signala
    fs, x = read('sounds/in.wav')
    time = np.arange(0, len(x) / fs, 1 / fs)
    time = time[:-1]

    # Racunanje FFT-a po segmentima
    freq, spek = segment_fft(x, fs)

    # Nalazenje frekvencije dominantne komponente
    dom_component = freq[np.argmax(spek)]

    # Zaustavni i propusni opseg
    ws = [.8 * dom_component, 1.2 * dom_component]
    wp = [.5 * dom_component, 1.6 * dom_component]

    # Filtriranje Butterwordom
    bb, ab = signal.butter(4, ws, btype='bandstop', analog=True)    
    b, a = signal.bilinear(bb, ab, fs / (2 * np.pi))

    x_butt = signal.lfilter(b, a, x)
    freq_elip, spek_butt   = segment_fft(x_butt, fs)
        
    write('sounds/out_butt.wav', fs, x_butt / max(x_butt))

    # Filtriranje Elipticnim
    be, ae = signal.ellip(4, 2, 40, ws, 'bandstop', analog=True)    
    b, a = signal.bilinear(bb, ae, fs / (2 * np.pi))

    # Filtriranje
    x_ellip = signal.lfilter(b, a, x)
    freq_elip, spek_ellip   = segment_fft(x_ellip, fs)
        
    write('sounds/out_elip.wav', fs, x_ellip / max(x_ellip))

    ## Plotovanje
    end_point = len(spek) // 5

    # a) signal vreme
    plt.figure()
    plt.plot(time, x)
    plt.xlabel('vreme [s]')
    plt.title('Snimak samoglasnika')

    plt.tight_layout()
    plt.savefig('figures/zad4_signal_vreme.png')

    # b) signal spektar 
    plt.figure()
    plt.stem(freq[:end_point], abs(spek[:end_point]), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Originalni spektar')

    plt.tight_layout()
    plt.savefig('figures/zad4_signal_spektar.png')

    # c) filteri
    plt.figure()

    plt.subplot(2, 1, 1)
    w, h = signal.freqs(bb, ab, np.logspace(0, 3, 500))
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterwordov filter')
    plt.xlabel('Frekvencija [rad / sec]')
    plt.ylabel('Amplituda [dB]')


    plt.subplot(2, 1, 2)
    w, h = signal.freqs(be, ae, np.logspace(0, 3, 500))
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Elipticni filter')
    plt.xlabel('Frekvencija [rad / sec]')
    plt.ylabel('Amplituda [dB]')

    plt.tight_layout()
    plt.savefig('figures/zad4_filteri.png')

    # d) filtrirani vreme 
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.xlabel('vreme [s]')
    plt.title('Nefiltrirani signal')
    plt.plot(time, x)
    
    plt.subplot(3, 1, 2)
    plt.plot(time, x_butt)
    plt.xlabel('vreme [s]')
    plt.title('Signal na izlazu Butterwordovog filtera')

    plt.subplot(3, 1, 3)
    plt.plot(time, x_ellip)
    plt.xlabel('vreme [s]')
    plt.title('Signal na izlazu Elipticnog filtera')

    plt.tight_layout()
    plt.savefig('figures/zad4_filtrirani_vreme.png')

    # e) AFK svih signala
    plt.figure()
    plt.tight_layout()

    plt.subplot(3, 1, 1)
    plt.stem(freq[:end_point], abs(spek[:end_point]), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Originalni spektar')

    plt.subplot(3, 1, 2)
    plt.stem(freq[:end_point], abs(spek_butt[:end_point]), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Spektar na izlazu Butterwordovog filtera')
    
    plt.subplot(3, 1, 3)
    plt.stem(freq[:end_point], abs(spek_ellip[:end_point]), markerfmt=',')
    plt.xlabel('Frekvancija [Hz]')
    plt.ylabel('FFT')
    plt.title('Spektar na izlazu Elipticnog filtera')

    plt.tight_layout()
    plt.savefig('figures/zad3_filtrirani_spektar.png')