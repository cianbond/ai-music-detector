# check_setup.py - quick test that all our libraries load correctly

import librosa
import soundfile
import numpy as np
import pandas as pd
import matplotlib
import sklearn

print("All libraries imported successfully!")
print("librosa version:", librosa.__version__)
print("numpy version:", np.__version__)
