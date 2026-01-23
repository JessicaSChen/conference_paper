import os
from pathlib import Path
import yt_dlp   # download Youtube video (background noise)
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.utils import ratio_to_db
import pyroomacoustics as pra
import numpy as np


BG_NOISE = ['https://www.youtube.com/watch?v=PYvac1EyIsY']
INPUT_ROOT = r"dataset"
OUTPUT_ROOT = r"output"

TARGET_SAMPLE_RATE = 16000 
TARGET_CHANNELS = 1

LEVELS = [5, 10, 20, 50, 100]    # % levels for signal and background noise levels

# Helper functions
def load_convert(path):
    """Load WAV and convert to mono and same sample rate and normalize audio."""
    audio = AudioSegment.from_wav(path)
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
    audio = audio.set_channels(TARGET_CHANNELS)
    audio = normalize(audio)
    return audio

def scale_audio(audio, percent):
    """Scale audio with percentage."""
    ratio = percent / 100.0
    gain_db = ratio_to_db(ratio)
    return audio.apply_gain(gain_db)

def mix_audios(speech, bgnoise, speech_pct, noise_pct):
    """Mix speech + noise at given percentages."""
    s = scale_audio(speech, speech_pct)
    n = scale_audio(bgnoise, noise_pct)

    # Trim bgnoise if shorter than speech
    if len(n) < len(s):
        loops = len(s) // len(n) + 1
        n = (n * loops)[:len(s)]
    else:
        n = n[:len(s)]

    return s.overlay(n)

def simulate_room(speech_audio: np.ndarray) -> np.ndarray:
    """
    Create a simulation of speech audio in a room
    """
    # 1. Define the room
    room_dim = [7.5, 7.5, 3]
    # the OR seems to have some echo/reverb according to https://www.youtube.com/watch?v=W7aRQGYhuk0
    room = pra.ShoeBox(room_dim, absorption=0.125, fs=TARGET_SAMPLE_RATE)

    # add raytracing
    room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)

    # NOTE: origin is in the bottom left corner!

    # 2. Source is right in the middle of the room
    source_pos = [3.75, 3.75, 1.5]
    # Speech
    room.add_source(source_pos, signal=speech_audio)

    # 3. Microphone in the top-left corner of the room (mimicing a surveillance camera position)
    mic_pos = [0, 7.5, 3]
    room.add_microphone_array(np.array([mic_pos]).T)

    # 4. Run the simulation
    room.simulate()

    # 5. The simulated audio (now quieter and with reverb)
    simulated_audio = room.mic_array.signals[0, :]

    return simulated_audio

def main():
    # Download Youtube background noise
    ydl_opts = { # yt_dlp options
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': ['-t', '600'], # first 10 mins download only
        'outtmpl': r'dataset/bg_noise', # output location
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

    print("Loaded background noise")

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

            # Mix audios to get every single audio and bg noise level combinations
            for noise_level, noise_path in noise_files.items():
                noise_audio = load_convert(noise_path)

                for s_pct in LEVELS:
                    for n_pct in LEVELS:
                        mixed = mix_audios(speech_audio, noise_audio, s_pct, n_pct)

                        out_name = f"{OUTPUT_ROOT}_audiovoice{s_pct}_bgnoise{n_pct}.wav"

                        out_path = os.path.join(speaker_output_dir, out_name)
                        mixed.export(out_path, format="wav")
                        print("Saved:", out_path)

    print("Output Audio Generated")

if __name__ == '__main__':
    main()