# ASR Evaluation in EVAR Surgical Workflow
This repository generates the final dataset used in the accompanying conference paper, First Experiences of Automated Speech Recognition on Different English Accents Spoken in Operating Rooms. The final dataset is derived from an input dataset of 12 English accents, is processed to support the ASR evaluation in EVAR surgical workflow.

Different Whisper, Wav2vec2, parakeet, and Canary ASR models were evaluted on these generated audio files. Refer to [ASR models](#asr-models) section for code used and produced results.
## Audio Generation 
### Requirements
- Python 3.9+
- `ffmpeg` installed and available on PATH
- Installed Python dependencies listed in `requirements.txt`

#### Install Requirements (FFmpeg)
##### Windows

Download and extract the `ffmpeg-release-essentials.zip` file from https://www.gyan.dev/ffmpeg/builds/.

##### macOS

Install using Homebrew: 
`brew install ffmpeg`

#### Install Dependencies
Execute the following command from the repository to install Python dependencies:
`pip install -r requirements.txt`

### Generate the Dataset
Execute the following command from the repository to run the script and generate the final dataset: 
`python -m audioGeneration`

### Outputs
The generated output files are saved and located in the `conference_paper/output` folder.

Each output file follows the following naming pattern, indicating the accent, gender, along with speech and noise levels: 
`<file_name>__audiovoice<speech_percent>_bgnoise<noise_percent>.wav`


## ASR models

### Whisper models
- Transcribe: `ASR_Models\whisper_transcription.py`
- Requirements: `ASR_Models\asr_requirements.txt`
- Model weights: [https://github.com/openai/whisper](https://github.com/openai/whisper)

### Wav2vec2 models
- Transcribe: `ASR_Models\hugging_face_transcription.py`
- Requirements: `ASR_Models\asr_requirements.txt`
- Model weights: [https://huggingface.co/facebook](https://huggingface.co/facebook)

### Nvidia parakeet and canary models
- Transcribe: `ASR_Models\nvidia_nemo_transcription.py`
- Requirements: `ASR_Models\nemo_requirements.txt`
- Model weights: [https://huggingface.co/nvidia](https://huggingface.co/nvidia)

### Baseline EVAR workflow text
- `ASR_Models\dataset\baseline_script.txt`

### ASR Results
- Each model has been evaluated three rounds on 350 audio files with varying speech and noise levels
- Transcription, accuracy, runtime details are present under `ASR_Results`

### Hardware used
- Intel Core Ultra 9 275HX x 24 cores
- CPU RAM: 64 GB
- GPU: RTX 5090 with CUDA 12.8
- GPU RAM: 24 GB
