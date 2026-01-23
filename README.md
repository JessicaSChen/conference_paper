# ASR Evaluation in EVAR Surgical Workflow
This repository generates the final dataset used in the accompanying conference paper, First Experiences of Automated Speech Recognition on Different English Accents Spoken in Operating Rooms. The final dataset is derived from an input dataset of 12 English accents, is processed to support the ASR evaluation in EVAR surgical workflow.

## Requirements
- Python 3.9+
- `ffmpeg` installed and available on PATH
- Installed Python dependencies listed in `requirements.txt`

### Install Requirements (FFmpeg)
#### Windows

Download and extract the `ffmpeg-release-essentials.zip` file from https://www.gyan.dev/ffmpeg/builds/.

#### macOS

Install using Homebrew: 
`brew install ffmpeg`

### Install Dependencies
Execute the following command from the repository to install Python dependencies:
`pip install -r requirements.txt`

## Generate the Dataset
Execute the following command from the repository to run the script and generate the final dataset: 
`python -m audioGeneration`

## Outputs
The generated output files are saved and located in the `conference_paper/output` folder.

Each output file follows the following naming pattern, indicating the accent, gender, along with speech and noise levels: 
`<file_name>__audiovoice<speech_percent>_bgnoise<noise_percent>.wav`

