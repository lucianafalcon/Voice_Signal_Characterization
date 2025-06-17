import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# === CARGA DE AUDIO ===
archivo = 'grabacion_lenta.wav'
ampl_y, Fs = librosa.load(archivo, sr=None)
duracion = len(ampl_y) / Fs
print(f"Duración del audio: {duracion:.2f} s")

# === ESPECTROGRAMA DE BANDA ANGOSTA ===
n_fft = 2048                   # Ventana larga para buena resolución en frecuencia
hop_length = 256              # Poco salto para continuidad temporal
ventana = np.hamming(n_fft)

D = librosa.stft(ampl_y, n_fft=n_fft, hop_length=hop_length, window=ventana)
D_db = librosa.amplitude_to_db(np.abs(D))

# === GRAFICAR EN ESCALA DE GRISES ===
plt.figure(figsize=(10, 5))
librosa.display.specshow(D_db, sr=Fs, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='gray')
plt.title('Espectrograma de Banda Angosta')
plt.colorbar(format='%+2.0f dB', label='Magnitud (dB)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.ylim([0, 4000])
plt.grid(lw=0.3)
plt.tight_layout()
plt.show()
