# 🎙️ Voice Signal Analysis / Análisis de Señal de Voz

🇬🇧 This repository contains a signal processing project focused on the spectral and temporal analysis of human voice using Python.  
📄 The full technical report is written in Spanish and available as a PDF file.

---

## 📁 Repository Structure

### 🔧 Scripts

| File | Description |
|------|-------------|
| `0_grabacion_voz.py` | Script to record a voice signal |
| `1_fft_completa_mejorada.py` | FFT analysis using multiple periods |
| `1_fft_segun_periodo_mejorada.py` | FFT analysis of a single vocal period |
| `1_señales_periodicas_noperiodicas_mejorada.py` | Identification of periodic and aperiodic phonemes |
| `2_TD-PSOLA.py` | TD-PSOLA algorithm to modify pitch |
| `2_espectograma_banda_ancha_angosta_mejorada.py` | STFT spectrograms (narrowband/wideband) |
| `3_TFCT_aumento_velocidad_señal_lenta_mejorada.py` | Speed-up using phase vocoder (STFT) |
| `3_TFCT_disminucion_velocidad_señal_rapida_mejorada.py` | Slow-down using STFT |
| `3_decimacion_señal_lenta_mejorada.py` | Speed-up via decimation (time domain) |
| `3_interpolacion_señal_rapida_mejorada.py` | Slow-down via interpolation (time domain) |

### 🎧 Audio Input and Output Files

| File | Description |
|------|-------------|
| `grabacion_lenta.wav` | Slow recording of the word "Picasso" |
| `grabacion_rapida.wav` | Fast recording of the word "Picasso" |
| `picasso_lenta_decimada.wav` | Time-compressed signal using decimation |
| `picasso_rapida_interpolada.wav` | Time-stretched signal using interpolation |
| `picasso_lenta_acelerada_TFCT.wav` | Time-compressed using STFT vocoder |
| `picasso_rapida_lenta_TFCT.wav` | Time-stretched using STFT vocoder |

---

## ⚙️ Requirements

- Python 3.x  
- Dependencies: `numpy`, `scipy`, `matplotlib`, `sounddevice`, `librosa`

Install dependencies with:

```bash
pip install numpy scipy matplotlib sounddevice librosa
```markdown

▶️ How to Run
Optional: record your own voice

bash
Copy
Edit
python 0_grabacion_voz.py
Run the analysis scripts
For example:

bash
Copy
Edit
python 1_fft_completa_mejorada.py
python 2_TD-PSOLA.py
python 3_TFCT_disminucion_velocidad_señal_rapida_mejorada.py
Output .wav files will be saved in the same directory.

📄 Report
The full project report is available here:
📎 Señales___Sistemas___TP.pdf (in Spanish)

🧠 Summary
This project explores various techniques for analyzing and transforming speech signals. It includes:

Spectral analysis using the Fourier Transform (FFT)

Short-Time Fourier Transform (STFT) for spectrograms

Time-domain and frequency-domain pitch and speed modifications

TD-PSOLA and Phase Vocoder techniques

The results demonstrate the value of spectral-temporal processing for speech transformation, pitch shifting, and intelligibility preservation in different use cases.

⬇️ Scroll down for the full README in Spanish.
Este repositorio contiene un proyecto de procesamiento de señales centrado en el análisis espectral y temporal de la voz humana.
