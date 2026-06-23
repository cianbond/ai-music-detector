from huggingface_hub import snapshot_download
import pandas as pd
import glob

snapshot_download(

    repo_id="awsaf49/sonics",

    repo_type="dataset",

    allow_patterns=["*.csv"],

    local_dir="data",

)

csv_paths = glob.glob("data/**/*.csv", recursive=True)

print("Downloaded:", csv_paths)

for path in csv_paths:

    df = pd.read_csv(path)

    print(path)

    print("  shape:", df.shape)

    print("  columns:", df.columns.tolist())
    
    print(df.head(3), "\n")