# ruffed_grouse_occupancy_2024
Automated detection of Ruffed grouse for in prep manuscript (Goldman et al, in prep. "Assessing the Influence of Forest Type and West Nile Virus Risk on Ruffed Grouse (Bonasa umbellus) Occupancy in Regenerating Timber Harvests").

This repository contains a sample analysis of automated detection of ruffed grouse (_Bonasa umbellus_) drumming displays in audio recordings using the method from [1]. Running the python script produces automated detections of ruffed grouse drumming from the sample audio recordings.

We provide two 120-second recordings for sample analysis. 

The algorithm correctly detects four ruffed grouse drumming events in the provided sample recordings: starting around 30 and 60 seconds in the April 15 recording and around 15 and 80 seconds into the April 16 recording. 

### Files:
- `data`: contains sample recordings, 120 seconds from one recorder on two consecutive days
- `results`: contains outputs of running the sample analysis, including ruffed grouse detections and summary of survey effort
    - `detection_summary_...csv` : table counting the number of detected drumming events per file
    - `..._rugr_detections.csv`: full output of the automated method including information on each detected drumming sequence (see below for column name meanings)

    - `..._sampling_effort.csv`: summary of number of seconds of audio analyzed per day per recorder

    Please see the OpenSoundscape [documentation](https://opensoundscape.org/en/latest/api/modules.html#opensoundscape.signal_processing.find_accel_sequences) and [original paper's demo repo](https://github.com/kitzeslab/ruffed_grouse_manuscript_2022) for further details on the methods and results. 
- `detect_ruffed_grouse_drumming.py`: python script used to run the automated detector on the sample data
- `detect_ruffed_grouse_drumming.log`: output of running the python script with `python detect_ruffed_grouse_drumming.py > detect_ruffed_grouse_drumming.log`
- `rugr_python_environment.yml`: Environment file that can be used to reproduce a working conda environment to run these scripts. Create the environment with `conda create -f rugr_python_environment.yml`

#### Column name meanings for _rugr_detections.csv files:
- sequence_y: strenght of the detected pulses
- sequence_t: times of the detected pulses relative to the start of the audi ofile
- window_start_t: time of the start of the 60 second analysis window
- seq_len: number of pulses in the detected sequence
- seq_start_time: time of the first detected pulse relative to start of audio file
- seq_end_time: time of the last detected pulse relative to start of audio file	
- seq_midpoint_time: (seq_start_time + seq_end_time )/2

### Usage:
The python script can be modified and re-used on other datasets. 

As written, the script assumes audio files have a .WAV extension and are organized into sub-folders by recorder with the name of the recorder as the folder name. 

Running the python script as prepared (`python detect_ruffed_grouse_drumming.py > detect_ruffed_grouse_drumming.log`) re-creates all of the files in the `/results` folder and logs output to the .log file.

Contact sam.lapp@pitt.edu with questions or for further assistance. 

### References and resources

Paper describing the automated method:
[1] Lapp, Samuel, et al. "Automated recognition of ruffed grouse drumming in field recordings." Wildlife Society Bulletin 47.1 (2023): e1395.
> https://wildlife.onlinelibrary.wiley.com/doi/full/10.1002/wsb.1395

Sample analysis from original paper:
> https://github.com/kitzeslab/ruffed_grouse_manuscript_2022

Help with Conda/Python enviornments:
> https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
