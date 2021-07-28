import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rtlsdr import RtlSdr

Fs = 2.048e6  # sample rate
f = 446006250  # center frequency (Hz)
f_corr = 75  # frequency correction (ppm)
N = 2048  # number of samples to read

# set up SDR
sdr = RtlSdr()

sdr.sample_rate = Fs
sdr.center_freq = f
sdr.freq_correction = f_corr
sdr.gain = "auto"

# set up pyplot
fig, ax = plt.subplots()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.ylim((-90, 0))
plt.grid(True)

# read samples
r = sdr.read_samples(N)

# process samples
PSD = (np.abs(np.fft.fft(r)) / N)**2
PSD_log = 10.0 * np.log10(PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

f = np.linspace(Fs / -2.0, Fs / 2.0, N)  # lazy method
line, = ax.plot(f, PSD_shifted)


def animate(i):
    # read new samples
    r = sdr.read_samples(N)
    # process new samples
    PSD = (np.abs(np.fft.fft(r)) / N)**2
    PSD_log = 10.0 * np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)
    line.set_ydata(PSD_shifted)  # update the data
    return line,


ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, save_count=50)

plt.show()
