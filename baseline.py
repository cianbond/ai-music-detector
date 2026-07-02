"""
baseline.py
Classical ML baseline (Random Forest / SVM) for the AI-Generated Music Detector.

Loads the split table, extracts summary audio features (MFCCs and spectral features)
for every track. Splits into train/val/test using the leakage-safe split column and
scales the features. Trains a Random Forest and an SVM and evaluates both on the
held out test set (accuracy, precision, recall, F1, and confusion matrices).

Run from the repo root with:  python baseline.py
"""

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def extract_features(file_path, clip_seconds=15):

    """Load an audio file, take a centred 15s clip and return a 1D feature vector
       of summary audio features (MFCCs and spectral features) Returns None if track is too short"""

    # load the audio at a fixed 16kHz so wav and mp3 are on equal footing (kills the format giveaway (this was a problem found earlier on when implementing))
    y, sr = librosa.load(file_path, sr=16000)

    clip_len = clip_seconds * sr # convert clip length to total number of samples

    if len(y) < clip_len: # skip if its too short

        return None
    
    # take the centred 15s clip (same logic as get_clip in the preprocessing)

    middle = len(y) // 2 # centre sample of the track

    half = clip_len // 2 # half a clip to put on both sides of the middle

    clip = y[middle - half : middle + half] # this slices out the centred clip

    # extract features from the clip

    # MFCCs this is 13 numbers that summarise the overall "shape" of the sound (timbre) one value per time frame. a (13, 469) grid
    mfccs = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13)

    # each MFCC changes over time so summarise each one by its mean and standard deviation
    mfcc_mean = np.mean(mfccs, axis=1)

    mfcc_std  = np.std(mfccs, axis=1)

    # another 2 extra classic features (also summarised to mean)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=clip, sr=sr)) # brightness of the sound

    zero_crossing     = np.mean(librosa.feature.zero_crossing_rate(y=clip)) # noisiness of the sound

    # stick them all together into one flat feature vector (values for the 13 means, 13 standard deviation and the 2 extras)
    features = np.concatenate([mfcc_mean, mfcc_std, [spectral_centroid, zero_crossing]])

    return features


def build_audio_path(filename):

    """Build the full path to an audio file from its CSV filename.
       Fakes are .mp3 in data/audio/fake and reals are .wav in data/audio/real."""
    

    # fake filenames start with "fake_" so use the fake folder + .mp3
    if filename.startswith("fake_"):

        return os.path.join("data/audio/fake", filename + ".mp3")
    
    # otherwise it's a real track so use the real folder + .wav
    else:
        
        return os.path.join("data/audio/real", filename + ".wav")


# load the split table (same CSV the preprocessing used)
splits = pd.read_csv("data/subset_splits.csv")

print("Loaded splits:", splits.shape)


# lists to collect features (X) and labels (y) as we go. x = feature rows and y = labels ( 0 for real and 1 for fake)
X = []
y = []

# also track which rows that were skipped like in preprocessing
skipped_missing = 0

skipped_short = 0

splits_kept = []  # "train"/"val"/"test" for each kept track


# loop over every track in the split table
for i, row in splits.iterrows():

    filename = row["filename"]
    
    label = row["target"]  # 0 = real 1 = fake

    split = row["split"]  # which set this track belongs to

    audio_path = build_audio_path(filename)   # reuse the path helper from preprocessing

    # skip if the audio file isn't on disk (failed download)
    if not os.path.exists(audio_path):

        skipped_missing += 1

        continue

    # extract the 28 features for this track
    features = extract_features(audio_path)

    # skip if it came back None (too short)
    if features is None:

        skipped_short += 1

        continue

    # keep this tracks features, label and split together (they stay lined up)
    X.append(features)

    y.append(label)

    splits_kept.append(split) # kept in lockstep with X and y

# final summary of the run
print(f"Collected {len(X)} feature rows, skipped {skipped_missing} missing, {skipped_short} too short")

print(f"splits_kept length: {len(splits_kept)}  (should match X)")


# convert the collected lists from above into numpy arrays (needed for sklearn and array filtering)

X = np.array(X) # shape (3722, 28) - 3722 tracks and 28 features each

y = np.array(y) # shape (3722,) - the labels

splits_kept = np.array(splits_kept) # shape (3722,) - the split each track is in

print("X shape:", X.shape)

print("y shape:", y.shape)

print("splits_kept shape:", splits_kept.shape)


# first see exactly what split labels there are (so our filters match them)
print("Split values present:", np.unique(splits_kept))

# make a True/False mask for each split (True wherever the track is in that split)
train_mask = splits_kept == "train"

val_mask   = splits_kept == "val"

test_mask  = splits_kept == "test"

# use each mask to pull out that split's rows from X and y (same mask keeps them aligned)
X_train, y_train = X[train_mask], y[train_mask]

X_val,   y_val   = X[val_mask],   y[val_mask]

X_test,  y_test  = X[test_mask],  y[test_mask]

# check the sizes (train should be biggest ~80%, val/test ~10% each, summing to 3722)
print("Train:", X_train.shape, y_train.shape)

print("Val:  ", X_val.shape,   y_val.shape)

print("Test: ", X_test.shape,  y_test.shape)


# check each split has a healthy mix of both classes (0 = real 1 = fake)

print("Train label counts:", np.bincount(y_train)) # [real count, fake count]

print("Val label counts:  ", np.bincount(y_val))

print("Test label counts: ", np.bincount(y_test))


# create the scaler (rescales each feature to mean 0 and standard deviation 1)
scaler = StandardScaler()

# fit the scaler on the training features only (learns each features mean and spread from train)
# fitting on test too would leak test info into training so can't do that.
scaler.fit(X_train)

# now transform all three sets using that same fitted scaler
X_train_scaled = scaler.transform(X_train)

X_val_scaled   = scaler.transform(X_val)

X_test_scaled  = scaler.transform(X_test)

# quick before/after on one feature to see the effect
print("Before scaling - feature 0 (first 5 train rows):", X_train[:5, 0])

print("After scaling  - feature 0 (first 5 train rows):", X_train_scaled[:5, 0])

print("\nScaled train mean (should be ~0):", X_train_scaled.mean().round(3))

print("Scaled train std  (should be ~1):", X_train_scaled.std().round(3))


# create the Random Forest model. 100 trees. fixed random_state for reproducibility
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# train it on the raw training features (trees don't need scaling as they are not distance based)
rf.fit(X_train, y_train)

print("Random Forest trained.")

print("Number of trees:", len(rf.estimators_))


# create the SVM model (same fixed random_state for reproducibility as Random forest)
svm = SVC(random_state=42)

# train it on the scaled training features (SVM is distance based so it needs scaling)
svm.fit(X_train_scaled, y_train)

print("SVM trained.")


# get each models predictions on the test set (tracks neither model has seen)
rf_preds  = rf.predict(X_test)           # RF uses raw test features

svm_preds = svm.predict(X_test_scaled)   # SVM uses scaled test features

# see the first 10 predictions vs the true answers
print("True labels: ", y_test[:10])

print("RF preds:    ", rf_preds[:10])

print("SVM preds:   ", svm_preds[:10])


# evaluate the random forest on the test set
print("=" * 50)

print("RANDOM FOREST RESULTS")

print("=" * 50)

print("Accuracy:", round(accuracy_score(y_test, rf_preds), 3)) # overall what it got correct

print(classification_report(y_test, rf_preds, target_names=["real", "fake"])) # precision/recall/F1 per class


# evaluate the SVM on the test set
print("=" * 50)

print("SVM RESULTS")

print("=" * 50)

print("Accuracy:", round(accuracy_score(y_test, svm_preds), 3))

print(classification_report(y_test, svm_preds, target_names=["real", "fake"]))


# confusion matrix for each model:
#  rows = true class columns = predicted class

# diagonal = correct off-diagonal = mistakes (and which way they went)
print("RANDOM FOREST confusion matrix:")

print("            pred_real  pred_fake")

rf_cm = confusion_matrix(y_test, rf_preds)

print(f"true_real     {rf_cm[0,0]:4d}      {rf_cm[0,1]:4d}") # correct reals | reals called fake

print(f"true_fake     {rf_cm[1,0]:4d}      {rf_cm[1,1]:4d}") # fakes called real | correct fakes

print("\nSVM confusion matrix:")

print("            pred_real  pred_fake")

svm_cm = confusion_matrix(y_test, svm_preds)

print(f"true_real     {svm_cm[0,0]:4d}      {svm_cm[0,1]:4d}")

print(f"true_fake     {svm_cm[1,0]:4d}      {svm_cm[1,1]:4d}")