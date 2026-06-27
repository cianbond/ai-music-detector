"""
download_fake.py
Downloads the AI (fake) tracks for our subset from the SONICS dataset on Hugging Face.

The fake audio is distributed as 10 zip archives (fake_songs/part_01.zip through to part_10.zip),
each containing around 5000 mp3 files named fake_songs/<filename>.mp3

This script:
  1. reads our subsets fake track filenames from subset_splits.csv

  2. downloads each zip part (resumable: hf_hub_download caches completed files)

  3. extracts only the files we need from each zip (skips the rest)

  4. saves them into data/audio/fake/

  5. deletes each downloaded zip after extracting to get rid of ones not needed and save disk space

Resumable: skips tracks already extracted so it's safe to stop and re-run.
"""

import os
import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download

# setup 
splits = pd.read_csv("data/subset_splits.csv") # loads the splits table

fakes = splits[splits["target"] == 1] #keeps only the fake tracks (target 1 = fake)

FAKE_DIR = "data/audio/fake" # output folder

os.makedirs(FAKE_DIR, exist_ok=True)

# The set of mp3 filenames (inside the zip) that we actually want.
# Our metadata 'filename' is like "fake_44156_udio_1" 
# inside the zip its "fake_songs/fake_44156_udio_1.mp3". So have to build the set of wanted zip entry names.
# using a set instead of a list here so I can quickly find a specific file if required
wanted = set(f"fake_songs/{name}.mp3" for name in fakes["filename"])

print(f"We need {len(wanted)} fake tracks for our subset.\n")

# The 10 zip parts in the dataset
# using :02d to make the number 2 digits so it matchs the real files otherwise an error would occur
ZIP_PARTS = [f"fake_songs/part_{i:02d}.zip" for i in range(1, 11)]

extracted = 0   # counter for how many of the tracks pulled out so far

for part in ZIP_PARTS:

    print(f"--- {part} ---")

    # 1. download this zip part (skips if already downloaded)
    print("  downloading (or using cached copy)...")

    
    zip_path = hf_hub_download( # where the downloaded zip will land

        repo_id = "awsaf49/sonics", #the dataset to get it from

        repo_type = "dataset",

        filename = part,
    )

    # 2. open the zip and extract only the files we still need
    print("  extracting the tracks needed from this part...")

    with zipfile.ZipFile(zip_path, "r") as z: # this uses the imported zipfile tool to open the zip

        names_in_zip = set(z.namelist()) # this gives the list of names from the zip as a set so we can easily find the ones wanted below

        
        to_get = wanted & names_in_zip # to get which of our wanted files are in this zip. It keeps the ones that are in both.

        for entry in to_get:

            # the filename without the "fake_songs/" folder prefix
            base = os.path.basename(entry)          # drops the folder in front and just ives us last part of the path for tidiness e.g. fake_44156_udio_1.mp3
            
            out_path = os.path.join(FAKE_DIR, base) # this is the save location

            # skip if we already have it (makes the script resumable)
            if os.path.exists(out_path):

                extracted += 1 # count it skip it and continue to next one

                continue # skips it and continues to next one

            # read the file out of the zip and write it to data/audio/fake/ (if we dont have it so it hasnt skipped above)
            with z.open(entry) as src, open(out_path, "wb") as dst: # opens one file in entry at a time and names it src. second half creates a new file to save to

                dst.write(src.read()) # reads the whole mp3 file and writes into the new file (dst)

            extracted += 1

    print(f"  done. total extracted so far: {extracted}/{len(wanted)}")

    # 3. delete the downloaded zip to free disk space. Whats needed is taken at this point so no need for all of the other tracks
    try:

        os.remove(zip_path) # deletes the zip as all the required files have already been copied to somewhere else

        print(f"  deleted {part} to free space.\n")

    except OSError as e:

        print(f"  (couldn't delete zip: {e})\n") # if theres an error for some reasonn mark it but don't crash

print(f"DONE. Extracted {extracted}/{len(wanted)} fake tracks into {FAKE_DIR}") # this only occurs at very end when all 10 zips are done

# report any that couldn't be found found (should be none if naming matches)
if extracted < len(wanted): # i.e. if extracted is less than 2000 then something must be missing

    print(f"WARNING: {len(wanted) - extracted} tracks were not found in the zips.")