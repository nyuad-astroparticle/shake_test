import pandas as pd
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os


def read_sdi_data(file_path, axis = 'X'):
    """
    This function is used to read the data files produced by sdi logger

    **Input**

    *file_path* is the sdi logger shake test output csv file
    *axis* default value 'X'. is the axis to focus on. renames that axis to 'Acceleration (g)'

    **Output**

    *data* is a pandas dataframe containing timestamps and X,Y,Z accelreation values
    
    """
    data = pd.read_csv(file_path)
    data.columns = ['Timestamp', 'X', 'Y', 'Z']
    data = data.rename(columns={axis:'Acceleration (g)'})

    # Convert the 'Timestamp' column to datetime objects
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%m/%d/%Y %H:%M:%S.%f')

    # Calculate the time in seconds relative to the first timestamp
    start_time = data['Timestamp'].iloc[0]
    data['Timestamp'] = (data['Timestamp'] - start_time).dt.total_seconds()

    return data


def read_input_data(file_path):
    
    """
    This file is to read the input psd spectrum defined by Domenico 
    """
    # Load the CSV file
    data = pd.read_csv(file_path, header=None)
    data.columns = ['Timestamp', 'Acceleration (g)']

    return data




def psd(data, sampling_rate = 10000):

    """
    This is a core function of this document. 

    **Input**

    *data* is the read functions data output

    *sampling_rate* is a sampling rate defined by the random noise tests sampling rate
    and the sdi logger sampling rate

    **Output**

    *binned_freqs* frequencies of the psd binned in log10 spectrum

    *binned_psd* g^/Hz psd values
    """
    # Extract relevant columns for plotting
    time_seconds = data['Timestamp']
    acceleration_g = data['Acceleration (g)']

    # Cleaning nan values
    acceleration_g = acceleration_g[~np.isnan(acceleration_g)]

    # Perform FFT on the acceleration data with the specified sampling rate
    fft_result = np.fft.fft(acceleration_g)
    frequencies = np.fft.fftfreq(len(acceleration_g), d=1/sampling_rate)

    # Calculate the Power Spectral Density (PSD) in g^2/Hz
    psd = (np.abs(fft_result) ** 2) / (len(acceleration_g) * sampling_rate)

    # Keep only the positive frequencies
    positive_freqs = frequencies[frequencies > 0]
    positive_psd = psd[frequencies > 0]

    # Define logarithmic bins for the frequencies
    num_bins = 50  # Number of bins (adjust as needed)
    log_bins = np.logspace(np.log10(min(positive_freqs)), np.log10(max(positive_freqs)), num_bins)

    # Bin the PSD values into the logarithmic bins
    binned_psd = []
    binned_freqs = []
    for i in range(len(log_bins) - 1):
        # Find indices within the current bin range
        bin_indices = (positive_freqs >= log_bins[i]) & (positive_freqs < log_bins[i + 1])
        if np.any(bin_indices):
            # Calculate the mean frequency and the mean PSD in this bin
            mean_freq = np.mean(positive_freqs[bin_indices])
            mean_psd = np.mean(positive_psd[bin_indices])
            binned_freqs.append(mean_freq)
            binned_psd.append(mean_psd)

    # Convert lists to numpy arrays for plotting
    binned_freqs = np.array(binned_freqs)
    binned_psd = np.array(binned_psd)
    return binned_freqs, binned_psd


def required_psd():

    """
    This function provides the linear sclae adjusted desired psd devided by 60

    **Input**

    *None*

    **Output**

    *fine_freq_linear* frequencies

    *fine_asd_linear* psd values

    """
   
    frequency = np.array([20, 100, 500, 2000])
    asd = np.array([0.18, 0.7, 0.7, 0.02])

    # Performing linear interpolation in log10 space
    log_freq = np.log10(frequency)
    log_asd = np.log10(asd)

    interpolator = interp1d(log_freq, log_asd, kind='linear')

    # Generating finer points for the interpolated curve
    fine_freq = np.linspace(log_freq.min(), log_freq.max(), 100)
    fine_asd = interpolator(fine_freq)

    # Converting back to linear space for both frequency and ASD
    fine_freq_linear = 10 ** fine_freq
    fine_asd_linear = 10 ** fine_asd / 60

    return fine_freq_linear, fine_asd_linear

def giveLogSpaceFarray(lowF, highF, N):
    """
    This function gives a list of frequenceis in log10 space for the sinesweep

    **Input**

    *lowF* is low end of the frequency domain
    *highF* is high end included in the spectrum
    *N* is the number of steps in the log10 space we need 

    **Output**

    *frequencies* is an array with frequency values

    We work with the following function:
    log10 f_i = log2 f_0 + step * i
    """

    step = np.log10(highF/lowF)/N

    frequencies = []
    for i in range(N + 1):
        log10f = np.log10(lowF) + step * i
        frequencies.append(10**log10f)
    
    return frequencies


def createPulseDF(data):
    """
    This function create a pulse shape out of sine wave packets to find
    the frequency durations easier

    **Input**

    *data* is a pandas dataframe containing sine sweep data

    **Output**

    *pulse* is a pandas dataframe containing time and sigma columns, where is sigma is
    a binned average deviation of accelration sine wave away from 1g
    
    """
    sigma = (data['Acceleration (g)'] - np.ones(len(data)))**2 * 100
    data['sigma'] = sigma

    bin_edges = np.arange(0, data.Timestamp.max() + 1/20, 1/20)
    data['binned'] = pd.cut(data['Timestamp'], bins = bin_edges)
    grouped = data.groupby('binned')['sigma'].mean().reset_index()
    grouped['bin_center'] = grouped['binned'].apply(lambda x: (x.left + x.right) / 2 if pd.notnull(x) else np.nan)
    grouped = grouped.drop(columns=['binned'])

    pulse = grouped
    pulse = pulse.rename(columns={'bin_center':'time'})
    pulse = pulse[['time', 'sigma']]
    
    return pulse

def findPulses(data, threshold = 0.02):

    """
    This function tries to find the boundaries for the pulses

    **Input** 

    *data* is a pulse dataframe

    **Output** 

    *pulse_starts* is an array of pulse starting points

    *pulse_ends* is an array of pulse ending points
    
    """

    # Identify pulse start and end points based on the condition (increase by 10 or more)
    duration_threshold = 1.076  # seconds

    # Correct logic to ensure no overlapping pulses (each start must have a corresponding end)
    pulse_starts = []
    pulse_ends = []
    in_pulse = False  # Track if currently in a pulse

    for i in range(1, len(data)):
        if not in_pulse and data['sigma'][i] - data['sigma'][i - 1] >= threshold:
            pulse_start_time = data['time'][i]
            pulse_starts.append(pulse_start_time)
            pulse_end_time = pulse_start_time + duration_threshold
            pulse_ends.append(pulse_end_time)
            in_pulse = True  # Mark as in a pulse
        if in_pulse and data['time'][i] >= pulse_end_time:
            in_pulse = False  # Reset pulse status after the end time

    return pulse_starts, pulse_ends



def findAmplitudes(pulse_starts, pulse_ends, data):
    """
    This function gives amplitude values computed using sinewave curvefit
    for maximum accuracy 

    **Input**

    *data* is the original sinesweep dataframe read using read_sdi_data

    *pulse_starts* check findPulses

    *pulse_ends* check findPulses

    **Output**
    
    """
    amplitudes = []

    for i in range(len(pulse_starts)):

        low = pulse_starts[i]
        high = pulse_ends[i]

        data = data.dropna()
        amplitude = data[(data.Timestamp > low) & (data.Timestamp < high)]
        omega = np.pi * 2 * giveLogSpaceFarray(20,2000,50)[i]

        def sine_wave(t, A, phi):
            return A * np.sin(omega * t + phi) + 1

        t = amplitude['Timestamp'].values
        y = amplitude['Acceleration (g)'].values
        initial_guess = [1, 0]

        params, covariance = curve_fit(sine_wave, t, y, p0=initial_guess)
        amplitude_value, phase = params

        if abs(amplitude_value) > 10: amplitude_value = 0
        amplitudes.append(abs(amplitude_value))

    return amplitudes


def checkFolder(folder_path):
    """
    Checks if a folder exists at the given path, and if not, creates it.

    **Intput**
        *folder_path* (str): The path to the folder.

    **Output**

        *None*
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")