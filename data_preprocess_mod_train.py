#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import numpy as np
from hparam import hparam as hp
from pyAudioAnalysis import audioSegmentation as aS
import python_speech_features as psf
import soundfile as sf

# This preprocess method is a modification from Harry Volek's original implementation
# The features extracted are from python_speech_features library and the array shapes are different
# The silence detection is done from a custom amde function which in turn uses the library pyAudioAnalysis


# downloaded dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))

def silence_detection(utter):
    intervals = aS.silence_removal(utter, 
                                   hp.data.sr, 
                                   0.025, 
                                   0.010, 
                                   smooth_window = 1.0, 
                                   weight = 0.3, 
                                   plot = False)

    for i in range(len(intervals)):
        intervals[i] = [int(stamp*hp.data.sr) for stamp in intervals[i]]

    return intervals

def save_spectrogram_tisv(task):
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
    """
    print("start text independent utterance feature extraction")
    if task == 'train':
        os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
        audio_path = glob.glob(os.path.dirname(hp.data.train_path_unprocessed))
    elif task == 'test':
        os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file
        audio_path = glob.glob(os.path.dirname(hp.data.test_path_unprocessed))


    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    # total_speaker_num = len(audio_path)
    # train_speaker_num= (total_speaker_num//10)*9           # split total data 90% train and 10% test
    # print("total speaker number : %d"%total_speaker_num)
    # print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))

    for i, folder in enumerate(audio_path):
        # i > 586 for vox1 and i > 2996 for vox2
        if i>2996:
            break
        try:
            print("%dth speaker processing..."%i)
            utterances_spec = []
            utter_count = 0
            for utter_name in os.listdir(folder):
                if utter_name[-4:] == '.wav':
                    utter_count += 1
                    if utter_count > 10:
                        break

                    utter_path = os.path.join(folder, utter_name)         # path of each utterance
                    utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio

                    intervals = silence_detection(utter)

                    bits_per_sample = sf.SoundFile(utter_path)
                    bits_per_sample = int(bits_per_sample.subtype[-2:])

                    # bits_per_sample = 16 # assuming that bits per sample is 16 for .m4a files

                    utter /= 2**(bits_per_sample-1)

                    for interval in intervals:
                        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                            utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                            S = psf.base.logfbank(utter_part, 
                                                 samplerate=hp.data.sr, 
                                                 winlen=hp.data.window,
                                                 winstep=hp.data.hop, 
                                                 nfilt=hp.data.nmels, 
                                                 nfft=hp.data.nfft, 
                                                 lowfreq=0, 
                                                 highfreq=None, 
                                                 preemph=0.97)
                            # S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                            #                       win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                            # S = np.abs(S) ** 2
                            # mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                            # S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                            # # utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                            # # utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance
                            utterances_spec.append(S)
            utterances_spec = np.vstack(utterances_spec)
            print(utterances_spec.shape)
            if task =='train':      # save spectrogram as numpy file
                np.save(os.path.join(hp.data.train_path, "speaker%d.npy"%i), utterances_spec)
            elif task == 'test':
                np.save(os.path.join(hp.data.test_path, "speaker%d.npy"%i), utterances_spec)
        except:
            continue


if __name__ == "__main__":
    save_spectrogram_tisv('train')
    # save_spectrogram_tisv('test')
