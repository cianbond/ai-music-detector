# imports the tools needed. snapshot_download to fetch files from Hugging Face. pandas to read the CSVs glob to find file
from huggingface_hub import snapshot_download
import pandas as pd
import glob

# download just the metadata CSVs from the SONICS dataset (the audio is skipped)
snapshot_download(

    repo_id="awsaf49/sonics",

    repo_type="dataset",

    allow_patterns=["*.csv"],

    local_dir="data",

)

# find every .csv file inside the data folder (** + recursive=True searches subfolders too)
csv_paths = glob.glob("data/**/*.csv", recursive=True)

print("Downloaded:", csv_paths)

# loop through each CSV and print a quick summary of it
for path in csv_paths:

    df = pd.read_csv(path)

    print(path)

    print("  shape:", df.shape)

    print("  columns:", df.columns.tolist())
    
    print(df.head(3), "\n")