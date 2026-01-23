import os
import yt_dlp # download Youtube video (background noise)
from pathlib import Path
from pydub import AudioSegment

BG_NOISE = ['https://www.youtube.com/watch?v=PYvac1EyIsY']
INPUT_ROOT = r"dataset"
OUTPUT_ROOT = r"output"

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

def load_convert(path):
    """Load WAV and convert to mono + target sample rate."""
    audio = AudioSegment.from_wav(path)
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
    audio = audio.set_channels(TARGET_CHANNELS)
    return audio

def mix_same_length(speech, noise):
    """Trim noise to match speech."""
    if len(noise) < len(speech):
        loops = len(speech) // len(noise) + 1
        noise = (noise * loops)[:len(speech)]
    else:
        noise = noise[:len(speech)]
    return speech.overlay(noise)

def main():

    # Download Youtube background noise
    # yt_dlp options
    ydl_opts = {
        'format': 'bestaudio/best',

        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],

        # first 10 mins download only
        'postprocessor_args': ['-t', '600'],

        'outtmpl': r'dataset/bg_noise',
    }

    # Download Youtube
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(BG_NOISE)

    # Make final output root folder
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Load background noise files
    noise_files = {
        Path(f).stem.split("_")[1]: os.path.join(INPUT_ROOT, f)
        for f in os.listdir(INPUT_ROOT)
        if f.endswith(".wav") and "bg_noise" in f
    }

    print("Loaded noise types:", noise_files.keys())

    # Loop through each speaker folder
    for speaker_folder in os.listdir(INPUT_ROOT):
        speaker_path = os.path.join(INPUT_ROOT, speaker_folder)
        if not os.path.isdir(speaker_path):
            continue

        # Create corresponding output folder
        speaker_output_dir = os.path.join(OUTPUT_ROOT, speaker_folder)
        os.makedirs(speaker_output_dir, exist_ok=True)

        # Loop through WAV files
        for wav_file in os.listdir(speaker_path):
            if not wav_file.endswith(".wav"):
                continue

            # Extract voice level 
            name_noext = Path(wav_file).stem
            voice_level = "".join(filter(str.isdigit, name_noext))

            # Load speech
            speech_path = os.path.join(speaker_path, wav_file)
            speech_audio = load_convert(speech_path)

            # Mix with every bg noise level
            for noise_level, noise_path in noise_files.items():
                noise_audio = load_convert(noise_path)

                mixed = mix_same_length(speech_audio, noise_audio)

                # Output filename format
                out_name = (
                    f"{speaker_folder}_audiovoice{voice_level}"
                    f"_bgnoise{noise_level}.wav"
                )

                out_path = os.path.join(speaker_output_dir, out_name)
                mixed.export(out_path, format="wav")

                print("Saved:", out_path)

    print("Output Audio Generated")

if __name__ == '__main__':
    main()