import numpy as np
import scipy.signal as signal
import sounddevice as sd

from rtlsdr import RtlSdr

Fs = 1024000  # sample rate
f = 446006250  # center frequency (Hz)
f_corr = 75  # frequency correction (ppm)
N = 8192000  # number of samples to read

# set up SDR
sdr = RtlSdr()

# configure device
sdr.sample_rate = Fs
sdr.center_freq = f
sdr.freq_correction = f_corr
sdr.gain = "auto"


# Read samples
print(10 * "=", f"Capturing started, freq: {f}", 10 * "=")
samples = sdr.read_samples(N)
print(10 * "=", "Capturing stopped", 10 * "=")

# Convert samples to a numpy array
x2 = np.array(samples).astype("complex64")

# An FM broadcast signal has a bandwidth of 200 kHz
f_bw = 10000
n_taps = 64
# Use Remez algorithm to design filter coefficients
lpf = signal.remez(n_taps, [0, f_bw, f_bw + (Fs / 2 - f_bw) / 4, Fs / 2], [1, 0], Hz=Fs)
x3 = signal.lfilter(lpf, 1.0, x2)

dec_rate = int(Fs / f_bw)
x4 = x3[0::dec_rate]
# Calculate the new sampling rate
Fs_y = Fs / dec_rate

### Polar discriminator
y5 = x4[1:] * np.conj(x4[:-1])
x5 = np.angle(y5)

# The de-emphasis filter
# Given a signal 'x5' (in a numpy array) with sampling rate Fs_y
d = Fs_y * 75e-6   # Calculate the # of samples to hit the -3dB point
x = np.exp(-1 / d)   # Calculate the decay between each sample
b = [1 - x]          # Create the filter coefficients
a = [1, -x]
x6 = signal.lfilter(b, a, x5)

audio_freq = 10000.0
dec_audio = int(Fs_y / audio_freq)
Fs_audio = int(Fs_y / dec_audio)

x7 = signal.decimate(x6, dec_audio)

# Scale audio to adjust volume
x7 *= 75000 / np.max(np.abs(x7))

# print sampling freq and save raw sound to file
# print(Fs_audio)
# # Save to file as 16-bit signed single-channel audio samples
# x7.astype("int16").tofile("wbfm-mono.raw")

# play sound
wav_wave = np.array(x7, dtype=np.int16)
sd.play(wav_wave, Fs_audio, blocking=True)
