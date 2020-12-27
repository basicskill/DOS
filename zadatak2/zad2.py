import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read

# Postavka zadatka
Q = 3
fs = 4400

if __name__ == "__main__":
    # Ucitavanje originalnog signala
    fs_original, x = read('sounds/audio' + str(Q) + '.wav')
    time_original = np.arange(0, len(x) / fs_original, 1 / fs_original)

    # Odabiranje signala zadatom frekvencijom
    x_sampled = x[::fs_original // fs]
    time_sampled  = np.arange(0, len(x_sampled) / fs, 1 / fs)

    # Prikaz vremenskih oblika odabranog i ne odabranog signala
    plt.subplot(2, 1, 1)
    plt.title('Rezultat odabiranja signala u vremenu')
    plt.xlabel('vreme [s]')
    plt.ylabel('Originalni signal')
    plt.plot(time_original, x / max(x))
    plt.subplot(2, 1, 2)
    plt.xlabel('vreme [s]')
    plt.ylabel('Odabirani signal signal')
    plt.plot(time_sampled, x_sampled / max(x_sampled))
    plt.savefig('figures/zad2_odabiranje.png')

    # Furijeova transformacija originalnog signala
    X = np.fft.fft(x) / len(x)
    X = X[range(len(x) // 2)]
    X = abs(X) / max(abs(X))

    # Niz frekvencija za Furijeovu transformaciju
    freq = np.arange(len(x) // 2) * fs_original / len(x)

    # Poslednji prikaz u bitnom delu spektra
    last_idx = np.where(freq <= 1500)[0][-1]

    # Uklanjanje manje bitnih delova spektra
    # radi lepseg plotovanja
    X[X < 0.15] = None

    # Prikaz spektra signala
    plt.figure()
    plt.title('Amplitudska frekvencijska karakteristika signala')
    plt.xlabel('Frekvencija [Hz]')
    plt.ylabel('Amplituda')
    plt.stem(freq[:last_idx], X[:last_idx])
    plt.savefig('figures/zad2_spektar.png')

    # Trazeni tonovi

    tones = ['D3', 'E3', 'F3', 'G3', 'A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5']
    freqs = np.array([146.83, 164.81, 174.61, 196.00, 220.00, 220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 587.33])  
    main_freq = [freq[idx] for idx, f in enumerate(X) if f > 0.8]

    print([tones[np.argmin(abs(freqs - f))] for f in main_freq])