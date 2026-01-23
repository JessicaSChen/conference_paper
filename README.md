# conference_paper
This repository generates the final dataset used in the conference paper from the input dataset of english accents (total of 12).

## Requirements
- Python 3.9+
- `ffmpeg` installed and available on PATH
- Installed dependencies from requirements.txt

### Install Requirements (FFmpeg)
1. Windows
Download and extract the `ffmpeg-release-essentials.zip` file from https://www.gyan.dev/ffmpeg/builds/.

2. macOS
`brew install ffmpeg`

## Install Dependencies
`pip install -r requirements.txt`

## Run the Script
`python -m audioGeneration`

### Outputs
The generated output files are located in the output folder under conference_paper. 

Each filename follows the following pattern: 
`<file_name>__audiovoice<speech_percent>_bgnoise<noise_percent>.wav`

