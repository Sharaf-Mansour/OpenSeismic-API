import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy import signal
import os
from matplotlib.patches import Rectangle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def highpass_filter(data, cutoff, fs, order=4):
    """
    Applies a highpass Butterworth filter to the input data.

    Parameters:
    - data: Input signal (1D array)
    - cutoff: Cutoff frequency for the highpass filter in Hz
    - fs: Sampling frequency in Hz
    - order: The order of the filter (default is 4)

    Returns:
    - filtered_data: Highpass-filtered signal (1D array)
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist

    # Design Butterworth highpass filter
    b, a = butter(order, normal_cutoff, btype='high')

    # Apply the filter using filtfilt to prevent phase distortion
    filtered_data = filtfilt(b, a, data)

    return filtered_data

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a bandpass Butterworth filter to the input data.

    Parameters:
    - data: Input signal (1D array)
    - lowcut: Low cutoff frequency for the bandpass filter in Hz
    - highcut: High cutoff frequency for the bandpass filter in Hz
    - fs: Sampling frequency in Hz
    - order: The order of the filter (default is 4)

    Returns:
    - filtered_data: Bandpass-filtered signal (1D array)
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    print(low,high)
    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter using filtfilt to prevent phase distortion
    filtered_data = filtfilt(b, a, data)

    return filtered_data

def get_filtered(sig,highpass_cut=1, bandpass_lowcut= .1,bandpass_highcut =1.5, band_filt = True,high_filt=True):

    # Apply the high-pass filter to remove frequencies below 0.5 Hz
    filtered_trace = signal.detrend(sig)
    filtered_trace = signal.medfilt(filtered_trace, kernel_size=21)
    if high_filt:
        filtered_trace = highpass_filter(sig, highpass_cut, fs=20)
    if band_filt:
        filtered_trace = bandpass_filter(sig, bandpass_lowcut, bandpass_highcut,fs=20)


    return filtered_trace


def get_spectrogram_and_bbox(number_trace,data):

    filtered_trace0 =  get_filtered(data.get_waveforms(number_trace)[0].T,
                                highpass_cut =.5,
                        #      bandpass_lowcut= .01,
                        #       bandpass_highcut =1,
                                band_filt=False)
    frequencies, times, Sxx_0 = signal.spectrogram(filtered_trace0, 20,nfft = 2000,nperseg=400 , noverlap = 200,scaling = 'density')

    fig, ax = plt.subplots(figsize=(12, 8),dpi=100)

    # Hide axes and spines
    ax.axis('off')

    # Plot the spectrogram
    pcm = ax.pcolormesh(times, frequencies, Sxx_0, cmap='viridis', shading='gouraud')

    plt.savefig(f'data/{number_trace}.png', dpi=100) # saving figure

    ymin, ymax = ax.get_ylim() # get y axis limits

    bbox_x = data.metadata.iloc[number_trace]['trace_P_spectral_start_arrival_sample'] / 20 # left x
    bbox_width = data.metadata.iloc[number_trace]['trace_S_spectral_end_arrival_sample'] / 20 - bbox_x # Event width
    bbox_y = ymin # top limit
    bbox_height = ymax - ymin # Hight (digure height)
    ax.add_patch(plt.Rectangle((bbox_x, bbox_y), bbox_width, bbox_height, fill=False, edgecolor='red', linewidth=2)) # Add bounding box
    # draw the bounding box

    fig.savefig(f'box/box{number_trace}.png', dpi=100) # save figure
     
    return "data filltered"