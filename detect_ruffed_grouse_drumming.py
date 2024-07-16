"""
Run automated detector from [1] to detect ruffed grouse drumming in audio files

Produces three file types: ruffed grouse drumming detections (`...rugr_detections.csv`),
detection summaries (detection_summary_...csv) and summary of ARU survey effort, 
reported as the number of seconds of audio analyzed for each recorder (row) and date (column)

This script was run using opensoundscape version 0.8.0. The rugr_python_environment.yml file in this folder can be used to create a Python environment that matches
the environment used to run this script. For instance, using the conda package manger run:
`conda create -f rugr_python_environment.yml`

This script was run by first activating the conda environment, then running
```bash
python detect_ruffed_grouse_drumming.py > detect_ruffed_grouse_drumming.log
```

You may see warnings produced by underlying packages while running the script,
but this does not mean the script has failed.

[1] Lapp, Samuel, et al. "Automated recognition of ruffed grouse drumming in field recordings." Wildlife Society Bulletin 47.1 (2023): e1395.
"""

import datetime

print(f"{datetime.datetime.now()}: Started script")

from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.signal_processing import detect_peak_sequence_cwt
from opensoundscape.audiomoth import audiomoth_start_time
import pytz

import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import ray
from time import time as timer
import librosa

#### -----------------------Parameters to Modify----------------------------------- ####

# select dataset and location to save results
dataset_name = "sample_dataset"
dataset = f"./data"
results_dir = f"./results"
Path(results_dir).mkdir(parents=True, exist_ok=True)

## analysis script options ##

# if True, checks for output file and skips analysis of audio if present:
skip_completed_files = False

# if True, use the ray package to parallelize computation with cpu multiprocessing
parallelize_with_ray = False
ray_num_cpus = 10

# select date ranges and times of day to include
# here, we includes audio files starting between 04:00 - 12:00 EDT
time_zone = pytz.timezone("US/Eastern")
time_ranges = [
    [datetime.time(4, 0, 0), datetime.time(12, 0, 0)],
    # can list multiple time periods to include
]
date_ranges = [
    [datetime.date(2020, 4, 1), datetime.date(2020, 5, 31)],
    [datetime.date(2021, 4, 1), datetime.date(2021, 5, 31)],
    [datetime.date(2022, 4, 1), datetime.date(2022, 5, 31)],
    # can list multiple date ranges to include
]

# location to save summary of results; used to check which files are already completed
summary_df_path = Path(f"{results_dir}/detection_summary_{datetime.datetime.now()}.csv")

print(f"{datetime.datetime.now()}: Ruffed grouse detection on {dataset_name}")

# parameters for Ruffed Grouse detection method: for RUGR, probably don't change these (same parameters as Lapp 2022 WSB)
sample_rate = 400  # resample audio to this sample rate
wavelet = "morl"
peak_threshold = 0.2
window_len = 60  # sec to analyze in one chunk
center_frequency = 50  # for cwt
peak_separation = 15 / 400  # min duration (sec) between peaks


#### ---------------------Don't modify code below this------------------------------ ####


# analysis function for one file
# assumes that audio is organized into folders by device, eg /dataset/device_1/audio.wav
@ray.remote
def analyze(path):
    results_path = (
        f"{results_dir}/{Path(path).parent.name}_{Path(path).stem}_rugr_detections.csv"
    )

    if Path(results_path).exists() and skip_completed_files:
        # don't re-anlyze if the output file already exists, instead just load the results to include in the summary
        results = pd.read_csv(results_path)
        results = results.set_index(results.columns[0])
        results.index.name = "index"

    else:  # method to detect ruffed grouse
        # load the audio file into and Audio object
        audio = Audio.from_file(path)

        # All parameters used here are the default parameters of this function in OpenSoundscape 0.7.0,
        # however we write them exiplicitly in case defaults change in a future OpenSoundscape package version
        results = detect_peak_sequence_cwt(
            audio,
            sample_rate=sample_rate,
            window_len=window_len,
            center_frequency=center_frequency,
            wavelet=wavelet,
            peak_threshold=peak_threshold,
            peak_separation=peak_separation,
            dt_range=[0.05, 0.8],
            dy_range=[-0.2, 0],
            d2y_range=[-0.05, 0.15],
            max_skip=3,
            duration_range=[1, 15],
            points_range=[9, 100],
            plot=False,
        )  # returns a dataframe summarizing all detected sequences
        results.index.name = "index"

        # write results to csv
        results.to_csv(results_path)

    # return number of detections
    return path, results_path, len(results)


# initialize ray with automatically detected num_workers
ray.init(num_cpus=ray_num_cpus, num_gpus=0, local_mode=~parallelize_with_ray)


if summary_df_path.exists():
    df = pd.read_csv(summary_df_path).set_index("file")
else:
    # get list of all files in the dataset
    # assumes audio files have .WAV extension and are in organized into sub-folders by recorder
    files = glob(f"{dataset}/*/*.WAV")
    df = pd.DataFrame({"file": files})
    print(f"dataset {dataset_name} has {len(df)} files")

    # select files starting between 04:00 - 12:00 EDT
    df["datetime_utc"] = df["file"].apply(lambda f: audiomoth_start_time(Path(f).name))
    df["datetime_edt"] = df["datetime_utc"].apply(lambda t: t.astimezone(time_zone))
    df["date"] = df["datetime_edt"].apply(lambda t: t.date())
    df["time"] = df["datetime_edt"].apply(lambda t: t.time())

    def in_range(x, r):
        if x >= r[0] and x <= r[1]:
            return True
        return False

    # filter to files in date range
    df = df[
        df["date"].apply(
            lambda t: max([in_range(t, date_range) for date_range in date_ranges])
        )
    ]
    print(f"Filtered dataset by date: now has {len(df)} files")

    # filter to files starting in at least one of the time_ranges
    # in this case, files starting between 04 and 12 EDT
    df = df[
        df["time"].apply(
            lambda t: max([in_range(t, time_range) for time_range in time_ranges])
        )
    ]
    print(f"Filtered dataset by start times: now has {len(df)} files")

    # calculate survey effort per date and recorder, save to table
    print(f"{datetime.datetime.now()}: calculating survey effort")
    df = df.set_index("file")
    df["recorder_name"] = [Path(f).parent.name for f in df.index]

    def duration(f):
        try:
            return int(librosa.get_duration(filename=f))
        except:
            return -1

    df["audio_seconds"] = [duration(f) for f in df.index]

    # remove files that librosa failed to open
    df = df[df["audio_seconds"] > 0]

    sampling_effort = pd.pivot_table(
        data=df,
        values="audio_seconds",
        index="card",
        columns="date",
        aggfunc=sum,
        fill_value=0,
    )
    sampling_effort.to_csv(
        f"{results_dir}/{dataset_name}_selected_files_sampling_effort.csv"
    )

    # prepare df of files for analysis
    df["n_detections"] = np.nan
    df["results_path"] = np.nan
    df.to_csv(summary_df_path)

# skip files that are already completed
files_to_analyze = df[df["results_path"].apply(lambda x: x != x)].index.values
print(f"Skipping completed files: {len(files_to_analyze)} files left to analyze")

# spin off a bunch of (parallelized) ray tasks and keep their nametags
ray_nametags = [analyze.remote(f) for f in files_to_analyze]

# start a timer for the analysis
print(f"{datetime.datetime.now()}: Running detections")
t0 = timer()

# get the results back from the ray tasks as they finnish
for idx, r in enumerate(ray_nametags):
    file, results_path, n_detections = ray.get(r)
    df.at[file, "n_detections"] = int(n_detections)
    df.at[file, "results_path"] = results_path

    # save progress in table
    df.to_csv(summary_df_path)

    print(
        f"{datetime.datetime.now()}: Progress: {idx+1}/{len(ray_nametags)} complete after {(timer()-t0)/60:0.2f} minutes"
    )

print(f"{datetime.datetime.now()}: Completed all files. Results in: {results_dir}")
