import os
import pandas as pd
import torch
from jiwer import wer
import string
import re
from nemo_text_processing.text_normalization.normalize import Normalizer
import time
import argparse
from datetime import datetime
from transformers import pipeline
import logging


if __name__=="__main__":
    # user configurations from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", default="./dataset", help="dataset directory path")
    # parser.add_argument("--model_path", "-m", default="./huggingface-models/models--facebook--wav2vec2-large-robust-ft-libri-960h/snapshots/5d28473cc25ef7b338c9f731fe55626c4b082f58",
    #                      help="Hugging face model path.")
    # parser.add_argument("--model_disp_name", "-name", default="huggingface_wav2vec2_large_libri_960h", help="Reference name to be used in output files.")
    parser.add_argument("--model_path", "-m", default="./huggingface-models/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
                        help="Hugging face model path.")
    parser.add_argument("--model_disp_name", "-name", default="huggingface_wav2vec2_base_960h", help="Reference name to be used in output files.")
    parser.add_argument("--outputs_path", "-o", default="./huggingface-outputs", help="output directory path")
    args = parser.parse_args()

    
    
    dataset_path = args.dataset_path
    model_path = args.model_path
    outputs_path = args.outputs_path
    model_disp_name = args.model_disp_name

    # logs
    log_file_path = os.path.join(outputs_path, f"{model_disp_name}_{datetime.now().isoformat()}.log")
    logging.basicConfig(level=logging.DEBUG, 
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])

    logging.info(f"Timestamp: {datetime.now().isoformat()}")
    logging.info(f"{dataset_path=}")
    logging.info(f"{model_path=}")
    logging.info(f"{outputs_path=}\n")

    # baseline text prep
    logging.info("Normalizing baseline text...")
    baseline_t1 = time.time()
    normalizer = Normalizer(input_case='lower_cased', lang='en')
    f = open(f"{dataset_path}/baseline_script.txt")
    reference = f.read()
    reference = normalizer.normalize(reference.lower())
    reference = re.sub(f"[{string.punctuation.replace('-', '')}]", "", reference)
    reference = reference.lower().strip()
    reference = reference.replace("millimeter", "mm").replace("millimeters", "mm").replace("mms", "mm").replace("-", " ").replace("dryseal", "dry seal")
    reference = re.sub(r"\s+", " ", reference)
    f.close()
    baseline_t2 = time.time()
    logging.info(f"Finished normalizing baseline text in: {(baseline_t2-baseline_t1):.6f} seconds.\n")

    # load model
    logging.info(f"Loading model: {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = pipeline("automatic-speech-recognition", model=model_path, device="cuda")
    
    # wav files
    wav_files = []
    for f in os.listdir(dataset_path):
        if f.endswith(".wav"):
            wav_files.append(f)
    wav_files.sort()
    total_files = len(wav_files)
    logging.info(f"Number of audio files = {total_files}")

    # transcribe audio files in dataset
    out_stats = []
    for idx, f in enumerate(wav_files):
        # audio file path
        audio_file_path = os.path.join(dataset_path, f)
        logging.info("-------------------------------------------------")
        logging.info(f"Transcribing {idx+1}/{total_files}: {audio_file_path} ...")

        # transcribe
        t1 = time.time()
        result = asr_model(audio_file_path)
        t2 =time.time()
        transcribe_run_time_s = t2-t1
        logging.info(f"Finished transcribing in: {transcribe_run_time_s:.6f} seconds.\n")

        # output transcription normalize
        logging.info("Normalizing output transcribed text...")
        out_text_t1 = time.time()
        transcription = result["text"]
        transcription = normalizer.normalize(transcription.lower())
        transcription = re.sub(f"[{string.punctuation.replace('-', '')}]", "", transcription)
        transcription = transcription.lower().strip()
        transcription = transcription.replace("millimeter", "mm").replace("millimeters", "mm").replace("mms", "mm").replace("-", " ").replace('–', " ")
        transcription = re.sub(r"\s+", " ", transcription)
        out_text_t2 = time.time()
        logging.info(f"Finished normalizing output text in: {(out_text_t2-out_text_t1):.6f} seconds.\n")

        row = {"model": model_disp_name,
               "audio": f,
               "accent": f.split("_")[0],
               "gender": f.split("_")[1],
               "audiovoice": f.split("_")[3],
               "bgnoise": f.split("_")[5].split(".")[0],
               "error": wer(reference, transcription),
               "transcription": transcription,
               "transcribe_run_time_s": transcribe_run_time_s
               }
        
        out_stats.append(row)

        # clear cuda cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


    # save outputs
    logging.info("Saving outputs...")
    df = pd.DataFrame(data=out_stats)
    df.to_csv(os.path.join(outputs_path, f"{model_disp_name}_{datetime.now().isoformat()}.csv"), index=False)
    logging.info(f"End Timestamp: {datetime.now().isoformat()}\n")
