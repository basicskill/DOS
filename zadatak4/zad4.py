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

def plot_filter(b, a, name='figures/filter.png'):

    plt.figure()
    w, h = signal.freqs(b, a, np.logspace(0, 3, 500))
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Elliptical highpass filter fit to constraints')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.savefig(name)


if __name__ == "__main__":
    # Ucitavanje originalnog signala
    fs, x = read('sounds/in.wav')
    time = np.arange(0, len(x) / fs, 1 / fs)
    time = time[:-1]

    freq, spek = segment_fft(x, fs)

    # Nalazenje frekvencije dominantne komponente
    dom_component = freq[np.argmax(spek)]

    ws = [.8 * dom_component, 1.2 * dom_component]
    wp = [.5 * dom_component, 1.6 * dom_component]

    # Filtriranje Butterwordom
    # N, Wn = signal.buttord(wp, ws, 2, 40, True)
    b, a = signal.butter(4, ws, btype='bandstop', analog=True)    
    plot_filter(b, a, 'figures/filter_butt.png')
    b, a = signal.bilinear(b, a, fs / (2 * np.pi))

    x_ellip = signal.lfilter(b, a, x)
    freq_elip, spek_elip   = segment_fft(x_ellip, fs)
        
    plt.subplot(2, 1, 1)
    plt.title('Butter')
    plt.plot(freq[:len(freq) // 2], spek[:len(spek) // 2])
    plt.subplot(2, 1, 2)
    plt.plot(freq_elip[:len(freq_elip) // 2], spek_elip[:len(spek_elip) // 2])
    plt.savefig('figures/testing_butt.png')

    write('sounds/out_butt.wav', fs, x_ellip / max(x_ellip))

    # Filtriranje Elipticnim
    b, a = signal.ellip(4, 2, 40, ws, 'bandstop', analog=True)    
    plot_filter(b, a, 'figures/filter_ellip.png')
    b, a = signal.bilinear(b, a, fs / (2 * np.pi))

    x_ellip = signal.lfilter(b, a, x)
    freq_elip, spek_elip   = segment_fft(x_ellip, fs)
        
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freq[:len(freq) // 2], spek[:len(spek) // 2])
    plt.subplot(2, 1, 2)
    plt.plot(freq_elip[:len(freq_elip) // 2], spek_elip[:len(spek_elip) // 2])
    plt.title('Ellip')
    plt.savefig('figures/testing_ellip.png')

    write('sounds/out_elip.wav', fs, x_ellip / max(x_ellip))




