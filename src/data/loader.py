import pandas as pd
import numpy as np
import os

def load_slice_timeseries(filepath):
    """
    Loads the NASA SLICE pic_info excel file.
    Because the file has no headers, I manually assign them based on the data structure.
    """
    if not os.path.exists(filepath):
        return None

    # Read the excel file without headers
    df = pd.read_excel(filepath, header=None, engine='xlrd')

    # Assign column names based on the visual inspection of the data.
    # A: Frame/Index, B: Time, C: Exposure/Shutter, D: Unknown, E/F/G: Pixel intensities.
    df.columns = ['frame_idx','time_str','exposure','sensor_mode','intensity_1','intensity_2','intensity_3']

    # Convert intensity columns to numeric (in case there are text artifacts).
    for col in ['intensity_1','intensity_2','intensity_3']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df