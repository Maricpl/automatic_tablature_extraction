import os
import csv
import yaml
import museval
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
from tablature_extraction.source_separation import SeparationHub


def find_medleydb_tracks():
    medleydb_tracks = []
    with open("data/musdb18/tracklist.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row[2] == "MedleyDB":
                medleydb_tracks.append(row[0])
    return medleydb_tracks


def get_track_stems(track_name):
    parts = track_name.split(" - ")
    artist = parts[0].replace(" ", "")
    title = parts[1].replace(" ", "")
    metadata_file = f"data/musdb18/medley_db_metadata/{artist}_{title}_METADATA.yaml"

    if not os.path.exists(metadata_file):
        return None

    with open(metadata_file, "r") as f:
        metadata = yaml.safe_load(f)

    stems = {}
    for stem_id, stem_info in metadata.get("stems", {}).items():
        instrument = stem_info.get("instrument", "").lower()
        stems[stem_id] = instrument
    return stems


def filter_tracks(tracks):
    ACCEPTABLE_STEMS = [
        "acoustic guitar",
        "auxiliary percussion",
        "bass drum",
        "beatboxing",
        "bongo",
        "cabasa",
        "castanet",
        "chimes",
        "choir",
        "claps",
        "clean electric guitar",
        "conga",
        "cowbell",
        "crowd",
        "cymbal",
        "darbuka",
        "distorted electric guitar",
        "doumbek",
        "drum machine",
        "drum set",
        "electric bass",
        "female rapper",
        "female screamer",
        "female singer",
        "female speaker",
        "glockenspiel",
        "gong",
        "gu",
        "guiro",
        "high hat",
        "kick drum",
        "lap steel guitar",
        "male rapper",
        "male screamer",
        "male singer",
        "male speaker",
        "maracas",
        "marimba",
        "rattle",
        "shaker",
        "sleigh bells",
        "snaps",
        "snare drum",
        "tabla",
        "tambourine",
        "timpani",
        "toms",
        "triangle",
        "vibraphone",
        "vocalists",
        "whistle",
        "xylophone",
        "fx/processed sound"
    ]
    filtered_tracks = []
    for track in tracks:
        stems = get_track_stems(track)
        wrong_file = False
        for stem in stems.values():
            if stem not in ACCEPTABLE_STEMS:
                wrong_file = True
        if wrong_file:
            continue
        filtered_tracks.append(track)
    return filtered_tracks


def load_stem(track_name, model_name=None):
    stem_path = f"data/musdb18/wav/test/{track_name}/other.wav"
    if not os.path.exists(stem_path):
        stem_path = f"data/musdb18/wav/train/{track_name}/other.wav"

    if model_name:
        if "ht_demucs_guitar" in model_name:
            stem_path = f"data/results/{model_name}/{track_name}/guitar.wav"
        else:
            stem_path = f"data/results/{model_name}/{track_name}/other.wav"

    audio, _ = sf.read(stem_path)
    return audio

def evaluate_track(track_name, model_name):
    print(track_name)
    reference_stem = np.array(load_stem(track_name))
    estimated_stem = np.array(load_stem(track_name, model_name))

    if estimated_stem.ndim == 1:
        estimated_stem = np.expand_dims(estimated_stem, axis=1)
        estimated_stem = np.concatenate([estimated_stem, estimated_stem], axis=1)

    # Reshape for museval: (n_sources, n_samples, n_channels)
    reference_stem = reference_stem[np.newaxis, ...]
    estimated_stem = estimated_stem[np.newaxis, ...]

    scores = museval.evaluate(reference_stem, estimated_stem)

    sdr, sir, sar, isr = scores
    return np.nanmedian(sdr), np.nanmedian(sir), np.nanmedian(sar), np.nanmedian(isr)


def rename_mixtures(root_dir):
    mixture_files_exist = any("mixture.wav" in filenames for _, _, filenames in os.walk(root_dir))
    if not mixture_files_exist:
        print("No mixture.wav files found to rename.")
        return

    for dirpath, _, filenames in os.walk(root_dir):
        if "mixture.wav" in filenames:
            dir_name = os.path.basename(dirpath)
            old_file = os.path.join(dirpath, "mixture.wav")
            new_file = os.path.join(dirpath, f"{dir_name}.wav")
            print(f"Renaming {old_file} to {new_file}")
            os.rename(old_file, new_file)


def main():
    rename_mixtures("data/musdb18/wav") # change name of mixtures.wav to keep format of how separation mdoels work and save files
    models = SeparationHub.get_available_models()

    medleydb_tracks = find_medleydb_tracks()
    filtered_tracks = filter_tracks(medleydb_tracks)

    # Print number of filtered tracks and their total length
    print(f"Number of filtered tracks: {len(filtered_tracks)}")
    total_length = 0.0
    for track in filtered_tracks:
        audio_file = f"data/musdb18/wav/test/{track}/{track}.wav"
        if not os.path.exists(audio_file):
            audio_file = f"data/musdb18/wav/train/{track}/{track}.wav"
        if os.path.exists(audio_file):
            try:
                audio_info = sf.info(audio_file)
                total_length += audio_info.duration
            except Exception as e:
                print(f"Could not read {audio_file}: {e}")
    print(f"Summed length of filtered tracks: {total_length:.2f} seconds ({total_length/60:.2f} min)")

    print("--- Running Inference ---")
    for model_name in models:
        print(f"Running inference for model: {model_name}")

        separator = SeparationHub(model_name=model_name)
        for track in tqdm(filtered_tracks, desc=f"Inference on {model_name}"):
            audio_file = f"data/musdb18/wav/test/{track}/{track}.wav"
            if not os.path.exists(audio_file):
                audio_file = f"data/musdb18/wav/train/{track}/{track}.wav"

            _ = separator.separate(audio_file)

    print("\n--- Running Evaluation ---")
    all_scores = []
    for model in models:
        print(f"Evaluating model: {model}")
        for track in tqdm(filtered_tracks, desc=f"Evaluating {model}"):
            sdr, sir, sar, isr = evaluate_track(track, model)
            all_scores.append({
                "model": model,
                "track": track,
                "sdr": sdr,
                "sir": sir,
                "sar": sar,
                "isr": isr,
            })

    df = pd.DataFrame(all_scores)

    # Create the directory if it doesn't exist
    output_dir = "data/results/evaluation_source_separation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw scores to CSV
    df.to_csv(os.path.join(output_dir, "scores.csv"), index=False)
    
    # Calculate and print mean scores
    mean_scores = df.groupby("model")[["sdr", "sir", "sar", "isr"]].mean()
    print("\n" + "="*20)
    print("Mean Scores")
    print("="*20)
    print(mean_scores)
    
    # Save mean scores to CSV
    mean_scores.to_csv(os.path.join(output_dir, "mean_scores.csv"))


if __name__ == "__main__":
    main()
