import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# 1. Cargar la señal lenta
x_lenta, fs = librosa.load("grabacion_lenta.wav", sr=None)

# 2. STFT (TFCT)
n_fft = 1024    # n_fft = ventana
hop_length = 256  # hop_length = solapam. entre ventanas 
X = librosa.stft(x_lenta, n_fft=n_fft, hop_length=hop_length)  # Transforma. al dominio tiempo-frecuencia 

# 3. Eliminar una de cada 2 columnas (acelera x2)
X_fast = X[:, ::2]

# 4. iSTFT
x_fast = librosa.istft(X_fast, hop_length=hop_length) 

# 5. Guardar la señal acelerada
sf.write("picasso_lenta_acelerada_TFCT.wav", x_fast, fs)

# 6. Espectrograma en escala de grises
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X_fast), ref=np.max),
                         sr=fs, hop_length=hop_length, y_axis='linear', x_axis='time', cmap='gray')
plt.title("Picasso lenta acelerada x2 (TFCT)")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.ylim(0, 4000) #<--- el espectograma daba hasta 20kz
plt.show()
