import os
import numpy as np
import pandas as pd
import jams
import mir_eval
from tablature_extraction.transcription import TrascriptionHub
import music_transcription.python.config as mt_config
import random

# --- Note-level (TDR) and onset-level metrics ---
def calculate_note_level_metrics(predicted_notes, gt_notes, onset_window=0.05):
    gt_notes_eval = [
        {"start_time": n[0], "end_time": n[1], "string": int(n[2]), "fret": int(n[3])}
        for n in gt_notes
        if 0 <= int(n[2]) < 6 and 0 <= int(n[3]) <= 20
    ]
    if not gt_notes_eval and not predicted_notes:
        return {"tdr_precision": 1.0, "tdr_recall": 1.0, "tdr_f1": 1.0}
    tp_tdr = 0
    matched_pred_indices = set()
    for gt_note in gt_notes_eval:
        for pred_idx, pred_note in enumerate(predicted_notes):
            if pred_idx in matched_pred_indices:
                continue
            onset_match = abs(gt_note["start_time"] - pred_note["start_time"]) <= onset_window
            string_match = gt_note["string"] == pred_note["string"]
            fret_match = gt_note["fret"] == pred_note["fret"]
            if onset_match and string_match and fret_match:
                tp_tdr += 1
                matched_pred_indices.add(pred_idx)
                break
    p_tdr = tp_tdr / len(predicted_notes) if predicted_notes else (1.0 if not gt_notes_eval else 0.0)
    r_tdr = tp_tdr / len(gt_notes_eval) if gt_notes_eval else (1.0 if not predicted_notes else 0.0)
    f1_tdr = 2 * p_tdr * r_tdr / (p_tdr + r_tdr) if (p_tdr + r_tdr) > 0 else 0.0
    return {"tdr_precision": p_tdr, "tdr_recall": r_tdr, "tdr_f1": f1_tdr}

def calculate_onset_event_metrics(predicted_notes, gt_notes, onset_window=0.05):
    gt_onsets_times = np.unique(np.array([n[0] for n in gt_notes]))
    pred_onsets_times = np.unique(np.array([n['start_time'] for n in predicted_notes]))
    if len(pred_onsets_times) == 0:
        if len(gt_onsets_times) == 0:
            return {"onset_precision_event": 1.0, "onset_recall_event": 1.0, "onset_f1_event": 1.0}
        else:
            return {"onset_precision_event": 0.0, "onset_recall_event": 0.0, "onset_f1_event": 0.0}
    ons_f1, ons_p, ons_r = mir_eval.onset.f_measure(gt_onsets_times, pred_onsets_times, window=onset_window)
    return {"onset_precision_event": ons_p, "onset_recall_event": ons_r, "onset_f1_event": ons_f1}

def extract_annotations_from_jams(jams_file_path):
    notes = []
    jam_data = jams.load(jams_file_path)
    note_midi_annotations = jam_data.search(namespace="note_midi")
    for annotation_obj in note_midi_annotations:
        if not (
            annotation_obj.annotation_metadata
            and hasattr(annotation_obj.annotation_metadata, "data_source")
        ):
            continue
        string_num_str = annotation_obj.annotation_metadata.data_source
        if isinstance(string_num_str, str) and string_num_str.isdigit():
            string_num = int(string_num_str)
        else:
            continue
        if string_num not in mt_config.OPEN_STRING_PITCHES_JAMS:
            continue
        open_string_pitch = mt_config.OPEN_STRING_PITCHES_JAMS[string_num]
        for obs in annotation_obj.data:
            onset_sec = float(obs.time)
            duration_sec = float(obs.duration)
            offset_sec = onset_sec + duration_sec
            pitch_midi = float(obs.value)
            fret_num = int(round(pitch_midi - open_string_pitch))
            if fret_num < 0:
                fret_num = 0
            notes.append((onset_sec, offset_sec, string_num, fret_num, pitch_midi))
    notes.sort(key=lambda x: x[0])
    return notes

def collect_audio_files(audio_dir, exts=(".wav", ".mp3")):
    files = []
    # Only use files from audio_hex-pickup_original subdirectory
    for root, _, filenames in os.walk(audio_dir):
        for fname in filenames:
            if fname.lower().endswith(exts):
                files.append(os.path.join(root, fname))

    # get 20% of dataset for evaluation
    # set static seed for reproducibility
    random.seed(42)
    files = random.sample(files, int(0.2 * len(files)))
    return files

def main():
    audio_dir = "data/guitarset/audio_mono-mic/"
    jams_dir = "data/guitarset/annotation"
    output_base = "data/results"

    audio_files = collect_audio_files(audio_dir)
    model_names = TrascriptionHub.get_available_models()

    print("--- Running Inference ---")
    error_count = {model_name: 0 for model_name in model_names}
    error_files = {model_name: [] for model_name in model_names}
    for model_name in model_names:
        print(f"Running inference for model: {model_name}")
        model = TrascriptionHub(model_name=model_name)
        for audio_path in audio_files:
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            out_model_name = "tayuya/" + model_name if model_name == "basic_pitch" else model_name
            out_file = os.path.join(output_base, out_model_name, f"{stem}_onsets_frets.npz")
            if os.path.exists(out_file):
                print(f"Skipping inference for {model_name} - {stem} (already exists)")
                continue
            try:
                _ = model.transcribe(audio_path)
                print(f"Generated result for {model_name} - {stem}")
            except Exception as e:
                if (isinstance(e, UnboundLocalError) and 'note_info' in str(e)) or \
                   ('note_info' in str(e) and 'UnboundLocalError' in str(type(e))):
                    error_count[model_name] += 1
                    error_files[model_name].append(stem)
                    #print(f"Tayuya error for {model_name} - {stem}, skipping.")
                else:
                    raise
    
    print("\nInference completed. Error counts:")
    for model_name, count in error_count.items():
        print(f"{model_name}: {count} files skipped due to error")
        
    print("\n--- Running Evaluation ---")
    all_scores = []
    for model_name in model_names:
        print(f"Evaluating model: {model_name}")
        for audio_path in audio_files:
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            if stem in error_files[model_name]:
                print(f"Skipping evaluation for {model_name} - {stem} due to error.")
                continue
            
            # jams path should be stem without last _{microphone} 01_Rock1-90-C#_comp_mic -> 01_Rock1-90-C#_comp
            jams_stem = "_".join(stem.split("_")[:-1])
            jams_path = os.path.join(jams_dir, f"{jams_stem}.jams")
            out_model_name = "tayuya/" + model_name if model_name == "basic_pitch" else model_name
            pred_npz = os.path.join(output_base, out_model_name, f"{stem}_onsets_frets.npz")
            print(jams_path, pred_npz)
            if not os.path.exists(jams_path) or not os.path.exists(pred_npz):
                print(f"Missing data for {model_name} - {stem}, skipping.")
                continue
            gt_notes = extract_annotations_from_jams(jams_path)
            pred = np.load(pred_npz)
            pred_onsets = pred['onsets']
            pred_frets = pred['frets']
            n_frames, n_strings = pred_onsets.shape
            # Convert predictions to note events
            pred_notes = []
            for s in range(n_strings):
                active = False
                for f in range(n_frames):
                    if pred_onsets[f, s] > 0.5:
                        fret = int(pred_frets[f, s])
                        if not active:
                            start = f * (mt_config.HOP_LENGTH / mt_config.SAMPLE_RATE)
                            active = True
                            pred_notes.append({
                                'start_time': start,
                                'end_time': start + (mt_config.HOP_LENGTH / mt_config.SAMPLE_RATE),
                                'string': s,
                                'fret': fret
                            })
                        else:
                            pred_notes[-1]['end_time'] = f * (mt_config.HOP_LENGTH / mt_config.SAMPLE_RATE)
                    else:
                        active = False
            # TDR metrics
            tdr_metrics = calculate_note_level_metrics(pred_notes, gt_notes)
            # Onset metrics
            onset_metrics = calculate_onset_event_metrics(pred_notes, gt_notes)
            all_scores.append({
                "model": model_name,
                "track": stem,
                **tdr_metrics,
                **onset_metrics,
            })
            print(f"{stem} {model_name}: TDR_F1={tdr_metrics['tdr_f1']:.3f}, Onset_F1={onset_metrics['onset_f1_event']:.3f}")

    df = pd.DataFrame(all_scores)
    output_dir = os.path.join(output_base, "evaluation_transcription")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "scores.csv"), index=False)
    # Compute and save mean scores for all relevant metrics
    metric_cols = [
        "tdr_precision", "tdr_recall", "tdr_f1",
        "onset_precision_event", "onset_recall_event", "onset_f1_event"
    ]
    mean_scores = df.groupby("model")[metric_cols].mean()
    print("\n" + "="*20)
    print("Mean Scores (per model)")
    print("="*20)
    print(mean_scores)
    mean_scores.to_csv(os.path.join(output_dir, "mean_scores.csv"))

if __name__ == "__main__":
    main()
