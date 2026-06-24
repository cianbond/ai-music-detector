"""
download_audio.py
Downloads audio for the REAL (human) tracks in subset_splits.csv from YouTube.
Resumable: skips tracks already downloaded. Logs failures and keeps going.
(AI tracks are handled separately.)
"""

import os
import pandas as pd
import yt_dlp

# setup
splits = pd.read_csv("data/subset_splits.csv")

reals = splits[splits["target"] == 0]

REAL_DIR = "data/audio/real"
os.makedirs(REAL_DIR, exist_ok=True)

FAIL_LOG = "data/failed_downloads.csv"


def download_one(youtube_id, filename):

    """Download one tracks audio as wav. Returns True on success, False on failure."""
    out_path = f"{REAL_DIR}/{filename}.wav"

    # Skip if we already have it (this is what makes the script resumable and prevents duplicates)
    if os.path.exists(out_path):
        return True

    url = f"https://www.youtube.com/watch?v={youtube_id}"

    options = {

        "format": "bestaudio/best", # this grabs best audio only stream (fall back to best overall)

        "outtmpl": f"{REAL_DIR}/{filename}.%(ext)s",

        "postprocessors": [{ # after download jobs

            "key": "FFmpegExtractAudio", # use FFmpeg to extract the audio

            "preferredcodec": "wav", # convert it to wav

        }],

        # these 2 settings below are to silence output to reduce noise, warnings etc. when running
        "quiet": True,

        "no_warnings": True,
    }


    try: # attempt the download

        with yt_dlp.YoutubeDL(options) as ydl: # create the yt-dlp downloader with the settings from above

            ydl.download([url])

        return True
    
    except Exception as e: # this is for if an error occurs

        # A dead/unavailable video lands here. Don't crash. report and move on.
        print(f"  FAILED {filename} ({youtube_id}): {str(e)[:80]}") #limit it to 80 characters so it doesn't go on too long

        return False


# main loop
total = len(reals) # count how many rows (number of tracks to download)

ok = 0 # this counter counts every successful download

failed = [] # this creates a list of the tracks that fail

print(f"Starting download of {total} real tracks into {REAL_DIR}\n")

# for loop to go through every real track one row at a time
for i, (_, row) in enumerate(reals.iterrows(), start=1):

    yt_id = row["youtube_id"] # pull this tracks YouTube ID out of the row

    fname = row["filename"]  # pull this tracks filename out of the row


    success = download_one(yt_id, fname) # run the download for this track. True = worked and False = failed

    if success: # if/else to add to success count and log failures

        ok += 1  # track worked so add one to the success count

    else:

        # track failed so record its details
        failed.append({"filename": fname, "youtube_id": yt_id})

    # progress update every 25 tracks
    if i % 25 == 0:

        print(f"[{i}/{total}]  ok: {ok}  failed: {len(failed)}")


#  Save a log of all failures
if failed:

    pd.DataFrame(failed).to_csv(FAIL_LOG, index=False)

# print the final summary
print(f"\nDONE. {ok}/{total} succeeded, {len(failed)} failed.")

if failed:

    print(f"Failed ones logged to {FAIL_LOG}")