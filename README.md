# AI-Generated Music Detector

A machine learning pipeline that classifies short audio clips as either
human made or AI-generated music.

Higher Diploma in Science in Computing (Artificial Intelligence and Machine Learning), NCI

## Overview
The detector turns each audio clip into a mel-spectrogram and classifies it with a
convolutional neural network (CNN) trained by transfer learning from a pretrained
image model. A classical machine learning baseline (Random Forest/SVM) is built alongside it for comparison.

Beyond a standard held out test set the model is evaluated for robustness. It is tested  against AI generators it never saw during training and also degraded audio. Grad-CAM is
used to visualise which regions of the spectrogram drive each prediction, giving the
model a degree of explainability and interpretability.

## Dataset
This project uses the SONICS dataset (Rahman et al., 2025 — arXiv:2408.14080).
- AI-generated tracks (Suno / Udio) are distributed directly as mp3 files.
- Real, human tracks are referenced by YouTube ID and downloaded with yt-dlp.
- Licence: CC BY-NC 4.0
- Hugging Face: https://huggingface.co/datasets/awsaf49/sonics

A balanced, artist disjoint train/validation/test split is built from the metadata
(see explore_data.ipynb). This groups by artist for real tracks and by source-song ID
for AI tracks to prevent data leakage.

FakeMusicCaps is used as a held out cross-generator test set. Generators the
model never encountered during training.

> The audio files themselves are not included in this repository. They are large
> and fully regenerable from the metadata. Run the download step to get them.

## Project structure
    ai-music-detector/
    ├── explore_data.ipynb      # data loading, exploration, train/val/test split
    ├── download_metadata.py    # downloads the SONICS metadata CSVs
    ├── check_setup.py          # environment / dependency check
    ├── requirements.txt        # Python dependencies
    └── data/
        └── subset_splits.csv   # the balanced, artist-disjoint split

## Setup
    git clone https://github.com/cianbond/ai-music-detector.git
    cd ai-music-detector
    conda create -n ai-music-detector python=3.11
    conda activate ai-music-detector
    pip install -r requirements.txt

## How to run
1. Download the audio (real tracks via yt-dlp, AI tracks from SONICS).
2. Preprocess the audio into mel-spectrograms.
3. Train the classical baseline.
4. Train the CNN (transfer learning). Run on Google Colab for GPU access.
5. Evaluate and run the robustness tests.  Generate Grad-CAM visualisations.

## Status
Data acquisition, exploration and a leakage safe artist-disjoint train/validation/
test split are complete. Audio download, preprocessing, the baseline model and the
CNN are in progress.

## Author
Cian Bond
Student ID 25115596 
National College of Ireland
