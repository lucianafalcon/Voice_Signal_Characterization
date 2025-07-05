#----------------------------------------------------------------------------------------
#               Algoritmo para acelerar velocidad de la señal lenta  
#                              por TFCT (matriz) 
# TFCT -> Decimación (elimino frames)-> iTFCT (reconstuyo la señal en el dom del tiempo)
#----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# Cargar la señal
x_lenta, fs = librosa.load("grabacion_lenta.wav", sr=None)

#-------------------------------------------------------
# TFCT
n_fft = 1024   
hop_length = 256  
X = librosa.stft(x_lenta, n_fft=n_fft, hop_length=hop_length)  # Transformada señal dom temporal x_lenta 

# Eliminar una de cada 2 columnas: DECIMACION
X_fast = X[:, ::2]

# iTFCT
x_fast = librosa.istft(X_fast, hop_length=hop_length) 
#--------------------------------------------------------

# Guardar la señal acelerada
sf.write("picasso_lenta_acelerada_TFCT.wav", x_fast, fs)

# Graficar
plt.figure(figsize=(10, 4)) 

librosa.display.specshow(librosa.amplitude_to_db(np.abs(X_fast), ref=np.max), sr=fs, hop_length=hop_length, y_axis='linear', x_axis='time', cmap='gray')

plt.title("Picasso lenta acelerada x2 (TFCT)")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
