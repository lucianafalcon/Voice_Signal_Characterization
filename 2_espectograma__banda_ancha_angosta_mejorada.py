#-----------------------------------------------------
#         Script para graficar el espectro
#             de banda ancha y angosta  
#-----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Cargar archivo de audio
archivo = 'grabacion_lenta.wav'
ampl_y, Fs = librosa.load(archivo, sr=None)
duracion = len(ampl_y) / Fs
print(f"Duración del audio: {duracion:.2f} s")
print(f"Frec de muestreo: {Fs:.2f} Hz")

# ----------------------------------------------------
# Parámetros del fragmento a analizar 
#t_inicio = 1.49  # en segundos                <---------- SI ESTOY EN LA SEÑAL COMPLETA COMENTO, Y ' (para las vocales)'
#t_fin = 1.53    # en segundos
#vocal = 'a'                                 #  <--------- ACORDATE

#t_inicio = 2.16     
#t_fin = 2.2    
#vocal = 'o'

t_inicio = 0.42 
t_fin = 0.47    
vocal = 'i'   #banda-ancha=512, hop=128   banda ang: 2048, hop=256

#------------------------------------------------------------------- (para las vocales)
i_inicio = int(t_inicio * Fs)  # convierto el tiempo (seg) en muestras
i_fin = int(t_fin * Fs)

# Recorte sobre la misma señal ampl_y
ampl_y = ampl_y[i_inicio:i_fin]

#-------------------------------------------------------------------
# Espectograma de banda ancha → tiene ventana temporal corta
n_fft_ancha = 512                     # n_fft tamaño d la ventana: elijo segun duracion=n_fft/Fs                    <----- 
hop_length_ancha = 128               # el 75 de solap es el mas cool, mas computos pero igual es corta la señal    <----- SALTO ENTRE VENTANAS :)
#hop_length_ancha = n_fft_ancha // 2   #(para tener el hop al 50%)
ventana_ancha = np.hamming(n_fft_ancha)

D_ancha = librosa.stft(ampl_y, n_fft=n_fft_ancha, hop_length=hop_length_ancha, window=ventana_ancha)  #librosa.stft es la funcion que calcula la fft 
D_db_ancha = librosa.amplitude_to_db(np.abs(D_ancha))     # me quedo con la magnitud, descarto la fase 
#--------------------------------------------------------------------
# Espectograma de banda angosta
n_fft_angosta = 2048                   # Ventana larga para buena resolución en frecuencia       <-----
hop_length_angosta = 256              # Poco salto para continuidad temporal                     <----- 
#hop_length_angosta = n_fft_angosta // 2   #(para tener el hop al 50%)
ventana_angosta = np.hamming(n_fft_angosta)     #Rectangular, Hanning, Blackman, Gausiana, Kaiser (hamming es la posta para formantes y armonicos) atenua los extremos por la formita, y mete menos ruido por los laterales bajos

D_angosta = librosa.stft(ampl_y, n_fft=n_fft_angosta, hop_length=hop_length_angosta, window=ventana_angosta)
D_db_angosta = librosa.amplitude_to_db(np.abs(D_angosta))
#-------------------------------------------------------------------

# Graficar
plt.figure(figsize=(10, 5))   # figura vacía

librosa.display.specshow(D_db_ancha, sr=Fs, hop_length=hop_length_ancha, x_axis='time', y_axis='hz', cmap='gray') #si no pongo sr=Fs, hop_length=hop el grafic no queda en t real, chequeado 
# grafico.  sr=Fs ->(sampling rate): indica cant de  muestras hay por seg en el audio orig
#           hop_length=hop_length ->  distancia entre ventanas
plt.title(f'Espectrograma de Banda Ancha /{vocal}/')


plt.figure(figsize=(10, 5))
librosa.display.specshow(D_db_angosta, sr=Fs, hop_length=hop_length_angosta, x_axis='time', y_axis='hz', cmap='gray')
plt.title(f'Espectrograma de Banda Angosta /{vocal}/')


plt.colorbar(format='%+2.0f dB', label='Magnitud (dB)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.ylim([0, 4000])
plt.grid(lw=0.3)
plt.tight_layout()
plt.show()

