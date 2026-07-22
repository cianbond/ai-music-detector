# Flask app: upload an audio track, run the CNN and return human made vs AI-generated
from flask import Flask, render_template, request
import tempfile, os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from preprocess import process_file  # reuse the exact preprocessing from the notebook pipeline

import psycopg2 # the bridge that lets Python talk to PostgreSQL
from dotenv import load_dotenv # reads the .env file so the password stays out of the code

load_dotenv() # load the variables from .env into the environment

# save one prediction as a row in the database
def save_prediction(filename, verdict, confidence):

    conn = psycopg2.connect( # open a connection to the music_detector database
        
        dbname="music_detector",

        user="postgres",

        password=os.getenv("DB_PASSWORD"), # pulled from .env, never hard coded here 

        host="localhost",

        port="5432"
    )

    cur = conn.cursor() # a cursor is the thing that actually runs SQL commands

    cur.execute( # run an INSERT: %s are safe placeholders psycopg2 fills in for us
        "INSERT INTO predictions (filename, verdict, confidence) VALUES (%s, %s, %s)",
        (filename, verdict, confidence)
    )

    conn.commit() # commit: actually save the row (without this, nothing is written)

    cur.close() # close the cursor

    conn.close() # close the connection

app = Flask(__name__)

# ---- load the model once at startup (not per request) ----
device = torch.device("cpu")  # this laptop has no GPU, and single-file inference is instant on CPU

# rebuild the exact same ResNet18 shape used in training
model = models.resnet18(weights=None)  # empty skeleton (weights come from the .pth)

model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 channel in (greyscale spectrogram)

model.fc = nn.Linear(model.fc.in_features, 1)  # 1 output (real vs fake)

# pour the trained weights into the skeleton
model.load_state_dict(torch.load("cnn_resnet18_best.pth", map_location=device))

model.eval()  # evaluation mode (no training behaviour)


@app.route("/")
def index():

    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():

    # grab the uploaded file from the form
    uploaded = request.files["audio"]

    # save it to a temporary file on disk so librosa can read it (keep the original extension)
    suffix = os.path.splitext(uploaded.filename)[1]  # e.g. ".wav" or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:

        uploaded.save(tmp.name)  # write the upload to that temp path

        tmp_path = tmp.name

    # turn the audio into a spectrogram (same pipeline as training: centred 15s clip, mel-spectrogram)
    spec = process_file(tmp_path)

    os.remove(tmp_path)  # tidy up the temp file now we have the spectrogram

    # guard: process_file returns None if the track was too short for a full clip
    if spec is None:

        return render_template("index.html", result="Track too short to analyse (needs at least 15 seconds).")

    # shape the spectrogram the way the model expects: (1, 1, 128, time)
    tensor = torch.tensor(spec).float().unsqueeze(0).unsqueeze(0).to(device)

    # run the model (no gradients needed as only predicting)
    with torch.no_grad():

        logit = model(tensor)  # raw score

        prob = torch.sigmoid(logit).item()  # squash to a 0..1 probability of "fake"

    # turn the probability into a verdict and a confidence percentage
    if prob >= 0.5:

        verdict = "AI-generated"

        confidence = prob * 100

    else:

        verdict = "Human-made"

        confidence = (1 - prob) * 100

    result = f"{verdict}  (confidence: {confidence:.1f}%)"

    save_prediction(uploaded.filename, verdict, confidence) # write this prediction to the database

    return render_template("index.html", result=result, filename=uploaded.filename)


if __name__ == "__main__":

    app.run(debug=True)