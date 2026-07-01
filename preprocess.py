"""
preprocess.py
Preprocessing pipeline for the AI-Generated Music Detector.

Loads each track listed in data/subset_splits.csv and takes a centred 15 second
clip. Converts it to a mel-spectrogram and saves it to data/spectrograms/ as
an .npy file. Missing files (failed downloads) and tracks that are too short for a full
clip are skipped and reported.

Run from the repo root with python preprocess.py
"""

import os
import numpy as np
import pandas as pd
import librosa


# load an audio file and return a centred 15s clip (or None + a warning if its too short)
def get_clip(file_path, clip_seconds=15):

    """Load an audio file and return a centred clip of clip_seconds length.
       Returns None (and warns) if the track is shorter than clip_seconds."""

    # load the audio of whatever file path was passed in (y = waveform, sr = sample rate)
    y, sr = librosa.load(file_path)

    # how many samples is the clip
    clip_len = clip_seconds * sr

    # if the track is too short for a full clip then skip it
    if len(y) < clip_len:

        print("SKIPPED (too short):", file_path, "-", round(len(y) / sr, 1), "s") # name the file and its length

        return None # hand back None = "no usable clip" This stops the function here.

    # otherwise take the centred middle clip (as track is long enough)
    middle = len(y) // 2 # centre sample of the track

    half = clip_len // 2 # half a clip taken either side of the middle

    clip = y[middle - half : middle + half] # slice out the centred clip (y itself stays unchanged)


    return clip # hand back the finished 15s clip


def clip_to_spectrogram(clip, sr=22050, n_mels=128):

    """Convert an audio clip into a mel-spectrogram in decibels.
       Returns the spectrogram as a 2D numpy array (n_mels x time frames)."""

    # make the mel-spectrogram from the clip (uses the sr and n_mels passed in)
    mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=n_mels)

    # convert to decibels (loudness on a log scale and peak as the 0 dB reference)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db # give back the spectrogram data (just numbers here as thats what the CNN model reads. no drawing required)


def process_file(file_path, clip_seconds=15):

    """Load a file then take a centred clip and return its mel-spectrogram.
       Returns None if the clip was skipped (e.g. track was too short)."""

    # step 1- get the clip
    clip = get_clip(file_path, clip_seconds)

    # if the clip was skipped (too short) skip this file too
    if clip is None:

        return None

    # step 2- turn the clip into a mel-spectrogram
    spec = clip_to_spectrogram(clip)

    return spec # hand back the mel-spectrogram


def build_audio_path(filename):

    """Build the full path to an audio file from its CSV filename.
       Fakes are .mp3 in data/audio/fake and reals are .wav in data/audio/real."""

    if filename.startswith("fake_"):

        return os.path.join("data/audio/fake", filename + ".mp3") # fake filenames start with "fake_" and require .mp3

    else:

        return os.path.join("data/audio/real", filename + ".wav") # otherwise it will be a real track which require .wav extension


# Run the pipeline over all of the tracks and save each spectrogram to disk

# load the split table (all the tracks with their filename, label and split)
splits = pd.read_csv("data/subset_splits.csv")

# make the output folder for spectrograms
os.makedirs("data/spectrograms", exist_ok=True)

# counter for saved missing and too short
saved = 0

skipped_missing = 0

skipped_short = 0

for i, row in splits.iterrows(): # loops over all of the rows

    filename = row["filename"] # pull the tracks filename from this row

    audio_path = build_audio_path(filename) # build the full path to the audio file

    # skip if the audio file isn't on disk (failed download)
    if not os.path.exists(audio_path): # not flips it so it is if it does not exist

        print("MISSING:", filename)

        skipped_missing += 1

        continue # jump straight to next file

    # run the pipeline: path to clip to spectrogram
    spec = process_file(audio_path)

    # skip if it came back None (too short)
    if spec is None:

        skipped_short += 1

        continue

    # save the spectrogram as a .npy file named after the track
    out_path = os.path.join("data/spectrograms", filename + ".npy")

    np.save(out_path, spec)

    saved += 1 # count it

# summary of the run from the counters
print(f"\nDone: saved {saved}, missing {skipped_missing}, too short {skipped_short}")
