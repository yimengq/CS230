#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-07-23 14:26


import librosa
import numpy as np
import os
import pyworld
from pprint import pprint
import librosa.display
import time

"This file consists in encrypting the audio data, to use them in the code. It's just a file full of functions for the main file"

"Loading the sound waves using LibROSA.load (output is an array). All the files are added to a list there"
"Librosa is a library made in Python to work with audio files (and extract waves): https://iq.opengenus.org/introduction-to-librosa/"
def load_wavs(wav_dir, sr):
    wavs = list()
    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr, mono=True)
        # wav = wav.astype(np.float64)
        wavs.append(wav)
    return wavs

"Here, they use the module pyWORLD to extract f0 - Pitch contour, sp - the harmonic spectral envelope, and ap - the aperiodicity. fs is certainly the sample frequency"
"PyWORLD is a vocoder. It's designed for speech sampled with f >= 16 kHz"
def world_decompose(wav, fs, frame_period=5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    
    "F0_ceil and f0_floor may be different in our case. Maybe try : f0_floor=self.minf0, f0_ceil=self.maxf0"
    f0, timeaxis = pyworld.harvest(
        wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)

    # Finding Spectogram
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)

    # Finding aperiodicity
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    # Use this in Ipython to see plot
    # librosa.display.specshow(np.log(sp).T,
    #                          sr=fs,
    #                          hop_length=int(0.001 * fs * frame_period),
    #                          x_axis="time",
    #                          y_axis="linear",
    #                          cmap="magma")
    # colorbar()
    return f0, timeaxis, sp, ap

"Encode the spectrogram into binary numbers (I don't know the base)"
def world_encode_spectral_envelop(sp, fs, dim=24):
    # Get Mel-Cepstral coefficients (MCEPs)
    # sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp


"Decompose and encode the previously loaded data using the functions above. They add everything into lists"
def world_encode_data(wave, fs, frame_period=5.0, coded_dim=24):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()
    for wav in wave:
        f0, timeaxis, sp, ap = world_decompose(wav=wav,
                                               fs=fs,
                                               frame_period=frame_period)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)
    return f0s, timeaxes, sps, aps, coded_sps


" A function that make an average and a standard deviation for the f0s of the data"
def logf0_statistics(f0s):
    # Note: np.ma.log() calculating log on masked array (for incomplete or invalid entries in array)
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()
    return log_f0s_mean, log_f0s_std

"Function that transpose the arrays in the imput list: lst"
def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


"Normalization of the coded spectrograms as Andrew told us! No variance shifting this way!"
def coded_sps_normalization_fit_transform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append(
            (coded_sp - coded_sps_mean) / coded_sps_std)
    return coded_sps_normalized, coded_sps_mean, coded_sps_std

"Padding of the array extracted with the first function - libROSA library"
def wav_padding(wav, sr, frame_period, multiple=4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) +
                                      1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right),
                        'constant', constant_values=0)

    return wav_padded

"Normalization using a Gaussian Normalization for pitch conversions. The formula used is the formula of a Gaussian curve"
def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    # Logarithm Gaussian Normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) /
                          std_log_src * std_log_target + mean_log_target)
    return f0_converted

"Simply decode the coded spectrogram with a sample frequency using pyWORLD"
def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp

"Synthetize the speech into a wave format using pyWORLD"
def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    wav = wav.astype(np.float32)
    return wav

"Simply randoming the data order"
def sample_train_data(dataset_A, dataset_B, n_frames=128):
    # Created Pytorch custom dataset instead
    
    " Creating randomed index lists of A and B of the same length"
    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    "Put in list arrays of same length extracted from the data A and B, randomly"
    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:, start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:, start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B

"Creating the training sets A = X, B = Y of their audio files"
if __name__ == '__main__':
    start_time = time.time()
    wavs = load_wavs("../data/vcc2016_training/SF1/", 16000)
    # pprint(wavs)

    f0, timeaxis, sp, ap = world_decompose(wavs[0], 16000, 5.0)
    print(f0.shape, timeaxis.shape, sp.shape, ap.shape)

    coded_sp = world_encode_spectral_envelop(sp, 16000, 24)
    print(coded_sp.shape)

    f0s, timeaxes, sps, aps, coded_sps = world_encode_data(wavs, 16000, 5, 24)
    # print(f0s)

    log_f0_mean, log_f0_std = logf0_statistics(f0s)
    # print(log_f0_mean)

    coded_sps_transposed = transpose_in_list(lst=coded_sps)
    # print(coded_sps_transposed)

    coded_sps_norm, coded_sps_mean, coded_sps_std = coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_transposed)
    print(
        "Total time for preprcessing-> {:.4f}".format(time.time() - start_time))

    print(len(coded_sps_norm), coded_sps_norm[0].shape)
    temp_A = np.random.randn(162, 24, 550)
    temp_B = np.random.randn(158, 24, 550)

    a, b = sample_train_data(temp_A, temp_B)
    print(a.shape, b.shape)
