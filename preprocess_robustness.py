"""
preprocess_robustness.py
FR-006 preprocessing: builds the width at 10 seconds for the SONICs tracks as this time matches the FakeMusicCaps tracks

Part 1: reclips the SONICS test reals to their middle 10 seconds.

Part 2: samples 200 fakes per FakeMusicCaps generator (seed 42) at 10 seconds.

Spectrograms are saved as .npy files for the robustness evaluation in Colab.

Run from the repo root with python preprocess_robustness.py
"""

import os
import numpy as np
import pandas as pd

from preprocess import process_file, build_audio_path # borrow the existing pipeline functions (doesn't need to run whole pipeline now thanks to the __main__ guard)


# the splits table tells us which tracks belong to the held out test set
splits = pd.read_csv("data/subset_splits.csv")

# test reals only: the split column picks the test set and the filename prefix picks the reals
test_reals = splits[(splits["split"] == "test") & (splits["filename"].str.startswith("real_"))]

print("Test reals found:", len(test_reals)) # 192 on paper (24 of them are failed downloads with no file on disk)

print(test_reals.head()) # first few rows as a sanity check

# keep only the tracks that actually sat the CNN's exam: the ones whose spectrogram exists on disk
# (failed downloads never got a spectrogram so this mirrors the notebooks ghost matching rule exactly)
on_disk = []

for f in test_reals["filename"]: # walk the register name by name

    spec_path = os.path.join("data/spectrograms", f + ".npy") # the pigeonhole where that tracks spectrogram would live

    on_disk.append(os.path.exists(spec_path)) # True if a file is really in the hole and False if its a ghost

test_reals = test_reals[on_disk] # keep only the rows whose verdict is True

print("Test reals with spectrograms:", len(test_reals)) # should be 168 now matching the CNN's test set exactly

# Part 1: reclip the test reals to their middle 10 seconds and save the spectrograms
out_dir = os.path.join("data", "spectrograms_10s", "reals") # the 10 second tracks folder lives beside the main spectrograms. not inside them

os.makedirs(out_dir, exist_ok=True) # make the folder (and any missing parents) if not already there

saved = 0

skipped = 0

for f in test_reals["filename"]: # one lap per register name

    audio_path = build_audio_path(f) # the address finder from preprocess.py (real_ names point at data/audio/real)

    spec = process_file(audio_path, clip_seconds=10) # the whole pipeline in one call but with the dial turned to 10 seconds

    if spec is None: # too short for even 10 seconds (should never fire here as these all managed 15)

        skipped += 1

        continue

    np.save(os.path.join(out_dir, f + ".npy"), spec) # save under the same filename in the new folder

    saved += 1

    if saved % 50 == 0: # just so can see whats happening during the run

        print("Saved", saved, "of", len(test_reals))

print(f"\nPart 1 done: saved {saved}, skipped {skipped}")

# verify one saved spectrogram has the expected 10 second shape
check = np.load(os.path.join(out_dir, test_reals["filename"].iloc[0] + ".npy"))

print("Sample shape:", check.shape) # expecting (128, 313)