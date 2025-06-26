from dataclasses import dataclass

import flatbuffers
import librosa
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.fft import dct, dctn

import biodcase_tiny.feature_extraction.feature_config_generated as feature_config
from biodcase_tiny.feature_extraction.nb_fft import gen_twiddle, dsps_fft2r_sc16_ansi, do_fft
from biodcase_tiny.feature_extraction.nb_log32 import log32, vec_log32
from biodcase_tiny.feature_extraction.nb_mel import _init_filter_bank_weights, FILTER_BANK_ALIGNMENT, \
    FILTER_BANK_CHANNEL_BLOCK_SIZE, filter_bank, FilterBankConstants

from biodcase_tiny.feature_extraction.nb_isqrt import vec_sqrt64
from biodcase_tiny.feature_extraction.nb_shift_scale import shift_scale_up, shift_scale_down


@dataclass
class FeatureConstants:
    window_scaling_bits: np.uint8
    mel_post_scaling_bits: np.uint8
    hanning_window: NDArray[np.int16]
    fft_twiddle: NDArray[np.int16]
    mel_constants: FilterBankConstants


def make_constants(win_samples, sample_rate,  window_scaling_bits,
                   mel_n_channels, mel_low_hz, mel_high_hz, mel_post_scaling_bits):
    mel_constants = _init_filter_bank_weights(
        win_samples // 2, sample_rate, FILTER_BANK_ALIGNMENT,
        FILTER_BANK_CHANNEL_BLOCK_SIZE, mel_n_channels,
        mel_low_hz, mel_high_hz
    )
    hanning = np.round(np.hanning(win_samples) * (2 ** window_scaling_bits)).astype(np.int16)
    sc_table = gen_twiddle(win_samples)
    return FeatureConstants(
        window_scaling_bits=window_scaling_bits,
        mel_post_scaling_bits=mel_post_scaling_bits,
        hanning_window=hanning,
        fft_twiddle=sc_table,
        mel_constants=mel_constants
    )

def hz_to_erb(f):
    return 21.4 * np.log10(4.37e-3 * f + 1)

def erb_to_hz(erb):
    return (10**(erb / 21.4) - 1) / 4.37e-3

def make_gammatone_filterbank(sample_rate, n_fft, n_filters=40, f_min=50, f_max=None):
    if f_max is None:
        f_max = sample_rate / 2
    fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2)
    erb_low = hz_to_erb(f_min)
    erb_high = hz_to_erb(f_max)
    erb_points = np.linspace(erb_low, erb_high, n_filters)
    center_freqs = erb_to_hz(erb_points)
    erb_bandwidths = 24.7 + center_freqs / 9.265
    filters = np.zeros((n_filters, len(fft_freqs)))
    for i, (fc, bw) in enumerate(zip(center_freqs, erb_bandwidths)):
        f = fft_freqs
        filters[i, :] = 1.0 / (1.0 + ((f - fc) / (bw / 2)) ** 2)
    filters /= np.maximum(filters.sum(axis=1, keepdims=True), 1e-10)
    return filters

def apply_gammatone_filterbank(power_spectrum, filterbank):
    return np.dot(filterbank, power_spectrum)


def convert_constants(c: FeatureConstants):
    builder = flatbuffers.Builder(0)

    hanning_window_offset = builder.CreateNumpyVector(c.hanning_window)
    fft_twiddle_offset = builder.CreateNumpyVector(c.fft_twiddle)
    channel_freq_offset = builder.CreateNumpyVector(c.mel_constants.ch_freq_starts)
    channel_weight_offset = builder.CreateNumpyVector(c.mel_constants.ch_weight_starts)
    channel_width_offset = builder.CreateNumpyVector(c.mel_constants.ch_widths)
    weights_offset = builder.CreateNumpyVector(c.mel_constants.weights)
    unweights_offset = builder.CreateNumpyVector(c.mel_constants.unweights)

    # Build FilterbankConfig
    feature_config.FilterbankConfigStart(builder)
    feature_config.FilterbankConfigAddFftStartIdx(builder, c.mel_constants.fft_start_index)
    feature_config.FilterbankConfigAddFftEndIdx(builder, c.mel_constants.fft_end_index)
    feature_config.FilterbankConfigAddWeights(builder, weights_offset)
    feature_config.FilterbankConfigAddUnweights(builder, unweights_offset)
    feature_config.FilterbankConfigAddNumChannels(builder, c.mel_constants.n_channels)
    feature_config.FilterbankConfigAddChannelFrequencyStarts(builder, channel_freq_offset)
    feature_config.FilterbankConfigAddChannelWeightStarts(builder, channel_weight_offset)
    feature_config.FilterbankConfigAddChannelWidths(builder, channel_width_offset)
    fb_config_offset = feature_config.FilterbankConfigEnd(builder)

    # Build FeatureConfig
    feature_config.FeatureConfigStart(builder)
    feature_config.FeatureConfigAddWindowScalingBits(builder, c.window_scaling_bits)
    feature_config.FeatureConfigAddMelPostScalingBits(builder, c.mel_post_scaling_bits)
    feature_config.FeatureConfigAddHanningWindow(builder, hanning_window_offset)
    feature_config.FeatureConfigAddFftTwiddle(builder, fft_twiddle_offset)
    feature_config.FeatureConfigAddFbConfig(builder, fb_config_offset)
    feature_config_offset = feature_config.FeatureConfigEnd(builder)
    builder.Finish(feature_config_offset)
    buf = builder.Output()
    return buf


def apply_hanning(w, hanning, window_scaling_bits) -> NDArray[np.int16]:
    hanned = np.multiply(w, hanning, dtype=np.int32) >> window_scaling_bits
    hanned_clipped = np.clip(hanned, a_min=np.iinfo(np.int16).min, a_max=np.iinfo(np.int16).max)
    if (hanned_clipped.max() == np.iinfo(np.int16).max): print("was clipped")
    return hanned_clipped.astype(np.int16)


def energy(fft_vals):
    return np.power(fft_vals[0::2].astype(np.int32), 2).astype(np.uint32) + np.power(fft_vals[1::2].astype(np.int32), 2).astype(np.uint32)


def process_window(w, hanning, mel_constants: FilterBankConstants, fft_twiddle, window_scaling_bits, mel_post_scaling_bits, inference_mode=False):
    w = w.copy()

    # LM Task: get 1st and 2nd GFCC
    hanned: NDArray[np.int16] = apply_hanning(w, hanning, window_scaling_bits)
    scaled_bits = shift_scale_up(hanned)
    fft = do_fft(hanned, fft_twiddle)
    rfft = fft[:len(fft)//2]

    fft_energy: NDArray[np.int32] = energy(rfft)

    # DIFF starts here?

    gammatone_filters = make_gammatone_filterbank(
        sample_rate = 16000,
        n_fft=len(rfft),
        n_filters=mel_constants.n_channels,
        f_min=8000,
        f_max=3000,
    )

    gammatone_energy = apply_gammatone_filterbank(fft_energy, gammatone_filters)
    gfccs = dct(np.log(gammatone_energy + 1e-8), type=2, norm='ortho')

    return gfccs[:2]

    fft_energy[:mel_constants.fft_start_index] = 0
    fft_energy[mel_constants.fft_end_index:] = 0
    mel_scaled: NDArray[np.uint64] = filter_bank(
        mel_constants,
        fft_energy
    )
    mel_sqrt: NDArray[np.uint64] = vec_sqrt64(mel_scaled)
    shift_scale_down(mel_sqrt, scaled_bits)
    mel_logged: NDArray[np.uint32] = vec_log32(mel_sqrt, 1 << mel_post_scaling_bits, 0)
    mel_logged[mel_logged == 65535] = 0
    # TODO: figure out scaling factors
    if inference_mode:
        # TODO: in the scope of the DCASE, this will never be hit
        mel_rescaled = mel_logged.astype(np.int8)
    else:
        mel_rescaled = mel_logged.astype(float)
    # return librosa.feature.delta(mel_rescaled)
    return mel_rescaled


def extract_dct_from_features(features_2d, m=5, n=5):
    """
    Apply 2D DCT to the feature array and extract the top-left 5x5 corner.
    """
    # Ensure the input is a 2D array (e.g., output of do_windows_fn)
    features_2d = np.asarray(features_2d)

    dct_matrix = dctn(features_2d, type=2, norm='ortho')
    return dct_matrix[:m, :n].flatten()


def do_autocorr(data, window_len, window_stride):
    
    # Step 1: Find the index of the max absolute value in the signal
    max_idx = np.argmax(np.abs(data))
    
    # Step 2: Center a window of length `window_len` around this index
    half_win = window_len // 2
    start = max(0, max_idx - half_win)
    end = start + window_len

    # Adjust if window exceeds array bounds
    if end > len(data):
        end = len(data)
        start = end - window_len
    ref_window = data[start:end]
    
    xcorr = []
    for i in range(0, len(data) - window_len + 1, window_stride):
        segment = data[i:i + window_len]
        # Dot product (i.e., unnormalized cross-correlation)
        corr = np.dot(ref_window, segment)
        # Normalize (optional but recommended)
        # norm = np.linalg.norm(ref_window) * np.linalg.norm(segment) + 1e-8
        xcorr.append(corr)
    b = [(x // 50) for x in xcorr]
    return np.array(b)

def periodogram_ratio(psd: np.ndarray, epsilon=1e-10) -> float:
    """
    Computes the periodogram ratio: max power / total power.

    Parameters:
    - psd: 1D array of power spectral density values
    - epsilon: small value to avoid division by zero

    Returns:
    - ratio: float in [0, 1], higher means one dominant frequency
    """
    max_power = np.max(psd)
    total_power = np.sum(psd) + epsilon
    return max_power / total_power


def gfcc1_features(data, window_len=30, fs=16000):
    data = data.copy()

    window = np.ones(len(data))
    nfft = len(data)
    
    win_signal = data * window
    scale = np.sum(window**2)

    fft_vals = np.fft.fft(win_signal, n=nfft)
    fft_half = fft_vals[:nfft//2]
    psd = (1 / (fs * scale)) * np.abs(fft_half) ** 2
    psd[1:-1] *= 2  # account for power in symmetric bins

    ratio = periodogram_ratio(psd) * 100

    # data = ((data.astype(np.int32) - np.min(data)) * 8192 // (np.max(data) - np.min(data))).astype(np.int32)

    # Step 1: Find the index of the max absolute value in the signal
    max_idx = np.argmax(np.abs(data))
    
    # Step 2: Center a window of length `window_len` around this index
    half_win = window_len // 2
    start = max(0, max_idx - half_win)
    end = start + window_len

    # Adjust if window exceeds array bounds
    if end > len(data):
        end = len(data)
        start = end - window_len
    ref_window = data[start:end]
    p = 0
    xcorr = []
    for i in range(0, len(data) - window_len + 1, 20):
        segment = data[i:i + window_len]
        corr = np.dot(ref_window, segment)
        # Normalize (optional but recommended)
        # norm = np.linalg.norm(ref_window) * np.linalg.norm(segment) + 1e-8
        xcorr.append(corr)
        p = p+1

    # xcorr = np.array(xcorr)
    print(p, ", ", len(data))

    if np.max(xcorr) == np.min(xcorr):
        xcorr = 0
    else:
        xcorr = (xcorr - np.min(xcorr)) * 100 / (np.max(xcorr) - np.min(xcorr))
    #xcorr function where max value is 100 and min = 0

    xcorr = np.mean(xcorr)

    # print("xcorr = ", xcorr, ", periodgram = ", ratio)
    return xcorr, ratio

def gfcc2_features(data):

    # data = ((data.astype(np.int32) - np.min(data)) * 8192 // (np.max(data) - np.min(data))).astype(np.int32)
    # normalized between 0 and 2^13
    # get zero crossing rate and form factor??

    sign_changes = np.sign(data[1:]) != np.sign(data[:-1])
    zcr = np.sum(sign_changes)

    # Normalize to percentage
    zcr_percent = 100 * zcr / len(data)

    rms = np.sqrt(np.mean(data ** 2))
    mean_abs = np.mean(np.abs(data)) + 1e-8  # avoid divide-by-zero
    form_fact =  rms * 100 / mean_abs
    
    return zcr_percent, form_fact