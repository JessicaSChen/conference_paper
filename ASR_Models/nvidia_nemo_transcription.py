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
import nemo.collections.asr as nemo_asr
import logging


if __name__=="__main__":
    # user configurations from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", default="./dataset", help="dataset directory path")
    
    # parser.add_argument("--model_path", "-m", default="nvidia-models/models--nvidia--parakeet-rnnt-1.1b/snapshots/a07b19e98a26c1873a3f2622c446a4a1ca6316cb/parakeet-rnnt-1.1b.nemo", help="Nvidia nemo model path.")
    # parser.add_argument("--model_disp_name", "-name", default="nvidia_parakeet_rnnt_1.1b", help="Reference name to be used in output files.")
    
    # parser.add_argument("--model_path", "-m", default="nvidia-models/models--nvidia--parakeet-rnnt-0.6b/snapshots/b8a90ce81e7b6d41486154c7b2d96935077806c8/parakeet-rnnt-0.6b.nemo", help="Nvidia nemo model path.")
    # parser.add_argument("--model_disp_name", "-name", default="nvidia_parakeet_rnnt_0.6b", help="Reference name to be used in output files.")
    
    # parser.add_argument("--model_path", "-m", default="nvidia-models/models--nvidia--parakeet-tdt-0.6b-v2/snapshots/48b630d20b000e5ad3735e5378a2d9bde3f80826/parakeet-tdt-0.6b-v2.nemo", help="Nvidia nemo model path.")
    # parser.add_argument("--model_disp_name", "-name", default="parakeet_tdt_0.6b_v2", help="Reference name to be used in output files.")

    # parser.add_argument("--model_path", "-m", default="nvidia-models/models--nvidia--canary-1b/snapshots/1698acf1700ed316ffce1cb42d79437c7e360cfa/canary-1b.nemo", help="Nvidia nemo model path.")
    # parser.add_argument("--model_disp_name", "-name", default="canary_1b", help="Reference name to be used in output files.")

    # parser.add_argument("--model_path", "-m", default="nvidia-models/models--nvidia--canary-1b-flash/snapshots/a9a55e0295e7dd50d0c8c2a19491900a0daf24f3/canary-1b-flash.nemo", help="Nvidia nemo model path.")
    # parser.add_argument("--model_disp_name", "-name", default="canary_1b_flash", help="Reference name to be used in output files.")

    parser.add_argument("--model_path", "-m", default="nvidia-models/models--nvidia--canary-1b-v2/snapshots/87bc52657add533cd0156b3fc1aef027280754bf/canary-1b-v2.nemo", help="Nvidia nemo model path.")
    parser.add_argument("--model_disp_name", "-name", default="canary_1b_v2", help="Reference name to be used in output files.")
    
    parser.add_argument("--outputs_path", "-o", default="./nvidia-nemo-outputs-canary-v2/nvidia-nemo-outputs-03", help="output directory path")
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
    if "parakeet" in model_path:
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=model_path)
    elif "canary" in model_path:
        asr_model = nemo_asr.models.EncDecMultiTaskModel.restore_from(restore_path=model_path)

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
        logging.info("----------------------------------------------------------")
        logging.info(f"Transcribing {idx+1}/{total_files}: {audio_file_path} ...")

        # transcribe
        t1 = time.time()
        result = asr_model.transcribe(audio_file_path)
        t2 =time.time()
        transcribe_run_time_s = t2-t1
        logging.info(f"Finished transcribing in: {transcribe_run_time_s:.6f} seconds.\n")

        # output transcription normalize
        logging.info("Normalizing output transcribed text...")
        out_text_t1 = time.time()
        transcription = result[0].text
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
