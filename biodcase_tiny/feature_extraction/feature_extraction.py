from dataclasses import dataclass

import flatbuffers
import librosa
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.fft import dctn

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


def fft_absfiltered(fft_complex):
    # returns array of fft magnitude filtered between low and high cut off frequencies

    if not np.all(np.isfinite(fft_complex)):
        print("fft_complex contains non-finite values!")

    f_low = 5000
    f_high = 8000
    f_samp = 16000
    w_len = 1024

    a = np.sqrt(np.power(fft_complex[0::2].astype(np.int32), 2).astype(np.uint32) + np.power(fft_complex[1::2].astype(np.int32), 2).astype(np.uint32)).astype(np.uint32)
    return a[(f_low * w_len // f_samp):(f_high* w_len // f_samp)]

def process_window(w, hanning, mel_constants: FilterBankConstants, fft_twiddle, window_scaling_bits, mel_post_scaling_bits, inference_mode=False):
    w = w.copy()

    # MS: Get flux 
    hanned: NDArray[np.int16] = apply_hanning(w, hanning, window_scaling_bits) # 1024 sized hanning filtered
    scaled_bits = shift_scale_up(hanned)
    fft = do_fft(hanned, fft_twiddle)
    rfft = fft[:len(fft)//2]            

    # print(np.size(rfft))
    return fft_absfiltered(rfft)


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