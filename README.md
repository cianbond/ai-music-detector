# AI-Generated Music Detector

A machine learning pipeline that classifies short audio clips as either human made or AI-generated music with a web application for running the model on uploaded tracks.

Higher Diploma in Science in Computing (Artificial Intelligence and Machine Learning), NCI

## Overview
The detector turns each audio clip into a mel-spectrogram and classifies it with a convolutional neural network (CNN) trained by transfer learning from a pretrained
image model. A classical machine learning baseline (Random Forest/SVM) is built alongside it for comparison.

Beyond a standard held out test set the model is evaluated for robustness in two ways. It is tested against five AI generators it never saw during training and also against degraded audio (noise, pitch shifting and compression) to see how well it holds up when the signal is altered. Grad-CAM is used to visualise which regions of the spectrogram drive each prediction giving the
model a degree of explainability and interpretability.

A Flask web application wraps the trained model so a user can upload a track and get a prediction in the browser. Each prediction is stored in a PostgreSQL database.

## Key results
The CNN reaches 99.7% accuracy on a held out test set of Suno and Udio tracks. A train vs test gap of 0.14% shows it did not memorise its training tracks though as the robustness results below show, it did specialise to the generators it was trained on.

On generators it never saw during training its recall drops to between 10% and 43%, which shows it learned features specific to Suno and Udio rather than a general notion of AI-generated music.

Under degraded audio (tested on a balanced sample of 40 test tracks) the model is fragile but it fails in two opposite directions. Noise and pitch shifting collapse F1 from 1.00 to 0.08 by pushing the model to classify almost everything as human made. Bandwidth reduction (an 8 kHz resample) pushes it the other way classifying almost everything as AI-generated: its F1 of 0.69 reflects only the fake class and masks an accuracy of 0.55 below chance for this sample. Rather than gently blurring the decision boundary each form of degradation shifts the models operating point sharply to one class or the other.

The Grad-CAM visualisations are consistent with this picture. On Suno and Udio tracks the models attention sits over consistent spectral regions whereas on unseen generators it is more scattered. Because each heat map is normalised to its own maximum, the maps show where attention falls and how concentrated it is rather than its absolute strength.

## Dataset
This project uses the SONICS dataset (Rahman et al., 2025 — arXiv:2408.14080).
- AI-generated tracks (Suno / Udio) are distributed directly as mp3 files.
- Real, human tracks are referenced by YouTube ID and downloaded with yt-dlp.
- Licence: CC BY-NC 4.0
- Hugging Face: https://huggingface.co/datasets/awsaf49/sonics

A balanced artist disjoint train/validation/test split is built from the metadata (see explore_data.ipynb). This groups by artist for real tracks and by source song ID
for AI tracks to prevent data leakage.

FakeMusicCaps is used as a held out cross generator test set. Generators the model never encountered during training.
- Licence: CC BY-NC 4.0
- Zenodo: https://zenodo.org/records/11061273

> The audio files themselves are not included in this repository. They are large and fully regenerable from the metadata. Run the download step to get them.

## Tech stack
- Python 3.11
- PyTorch and torchvision (the CNN)
- librosa (audio to mel-spectrogram)
- scikit-learn (the classical baseline)
- Flask (the web application)
- PostgreSQL with psycopg2 (prediction storage)

## Project structure
    ai-music-detector/
    ├── explore_data.ipynb        # data loading, exploration, train/val/test split
    ├── download_metadata.py      # downloads the SONICS metadata CSVs
    ├── download_audio.py         # downloads the real tracks with yt-dlp
    ├── download_fake.py          # downloads / prepares the AI-generated tracks
    ├── preprocess.py             # turns audio into mel spectrograms
    ├── baseline.py               # the classical Random Forest / SVM baseline
    ├── cnn_training.ipynb        # CNN training and evaluation (transfer learning)
    ├── robustness_eval.ipynb     # cross-generator robustness tests and Grad-CAM
    ├── degraded_audio_eval.ipynb # degraded-audio robustness tests (noise, pitch, compression)
    ├── app.py                    # the Flask web application
    ├── templates/
    │   └── index.html            # the upload page
    ├── check_setup.py            # environment / dependency check
    ├── requirements.txt          # Python dependencies
    ├── .env.example              # template for the database password
    └── data/
        └── subset_splits.csv     # the balanced artist-disjoint split

## Setup
    git clone https://github.com/cianbond/ai-music-detector.git
    cd ai-music-detector
    conda create -n ai-music-detector python=3.11
    conda activate ai-music-detector
    pip install -r requirements.txt

## Running the web application
The web application needs PostgreSQL installed and running and the trained model file.

1. Install PostgreSQL and create the database and table:

        CREATE DATABASE music_detector;
        \c music_detector
        CREATE TABLE predictions (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );

2. Copy .env.example to .env and set your PostgreSQL password.

3. Place the trained model file (cnn_resnet18_best.pth) in the project root.

4. Run the app:

        python app.py

5. Open http://127.0.0.1:5000 in a browser and upload a track.

> The model file is not included in this repository because of its size. It is produced by running cnn_training.ipynb.

## Reproducing the pipeline
1. Download the audio (real tracks via yt-dlp, AI tracks from SONICS).

2. Preprocess the audio into mel spectrograms.

3. Train the classical baseline.

4. Train the CNN (transfer learning). Run on Google Colab for GPU access.

5. Evaluate and run the robustness tests. Generate Grad-CAM and degraded audio results.

## Author
Cian Bond
Student ID 25115596 
National College of Ireland
