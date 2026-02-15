from typing import Dict
import os
import numpy as np
import pandas as pd
import jams
import mir_eval
from tablature_extraction.transcription import TrascriptionHub
import music_transcription.python.config as mt_config
import random


def notes_to_fret_matrix(gt_notes, n_frames, n_strings, hop_length, sample_rate, max_frets, silence_fret_idx):
    """
    Convert list of note events to frame-wise [n_frames, n_strings] fret matrix for MPE metric.
    Each cell: fret number if note active, else silence_fret_idx.
    """
    gt_frets = np.full((n_frames, n_strings), silence_fret_idx, dtype=int)
    for note in gt_notes:
        onset, offset, string, fret, _ = note
        start_frame = int(np.round(onset * sample_rate / hop_length))
        end_frame = int(np.round(offset * sample_rate / hop_length))
        if 0 <= string < n_strings and 0 <= fret <= max_frets:
            gt_frets[start_frame:end_frame+1, string] = int(fret)
    return gt_frets

def calculate_mpe_metrics(pred_frets: np.ndarray, gt_frets: np.ndarray, silence_fret_idx: int, fret_padding_value: int = -100) -> Dict[str, float]:
    """
    Multi Pitch Estimation (MPE) metrics: frame-level F1, precision, recall.
    Args:
        pred_frets: np.ndarray [frames, strings] - predicted fret numbers
        gt_frets: np.ndarray [frames, strings] - ground truth fret numbers
        silence_fret_idx: int - value for silence class
        fret_padding_value: int - value for padding (ignored)
    Returns:
        Dict with mpe_f1, mpe_precision, mpe_recall
    """
    gt_active_mask = (gt_frets != silence_fret_idx) & (gt_frets != fret_padding_value)
    pred_active_mask = pred_frets != silence_fret_idx

    tp = ((gt_active_mask & pred_active_mask) & (gt_frets == pred_frets)).sum()
    fp = (pred_active_mask & ~gt_active_mask).sum() + ((gt_active_mask & pred_active_mask) & (gt_frets != pred_frets)).sum()
    fn = (~pred_active_mask & gt_active_mask).sum() + ((gt_active_mask & pred_active_mask) & (gt_frets != pred_frets)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "mpe_precision": float(precision),
        "mpe_recall": float(recall),
        "mpe_f1": float(f1),
    }

def calculate_note_level_metrics(predicted_notes, gt_notes, onset_window=0.05):
    gt_notes_eval = [
        {"start_time": n[0], "end_time": n[1], "string": int(n[2]), "fret": int(n[3]), "pitch": int(round(n[4]))}
        for n in gt_notes
        if 0 <= int(n[2]) < 6 and 0 <= int(n[3]) <= 20
    ]
    pred_notes_eval = [
        {"start_time": n["start_time"], "end_time": n["end_time"], "string": n["string"], "fret": n["fret"]}
        for n in predicted_notes
    ]

    for n in pred_notes_eval:
        # Estimate pitch from string and fret using GuitarSet tuning (EADGBE)
        if hasattr(mt_config, "OPEN_STRING_PITCHES_JAMS"):
            open_pitch = mt_config.OPEN_STRING_PITCHES_JAMS.get(n["string"], 40)
        else:
            open_pitch = 40 + 5 * n["string"]  # fallback: E2=40, A2=45, etc.
        n["pitch"] = int(round(open_pitch + n["fret"]))

    if not gt_notes_eval and not pred_notes_eval:
        return {"tdr": 1.0, "tab_precision": 1.0, "tab_recall": 1.0, "tab_f1": 1.0}

    # Step 1: Find all correct pitch matches (onset+pitch)
    matched_pred_indices_pitch = set()
    matched_gt_indices_pitch = set()
    for gt_idx, gt_note in enumerate(gt_notes_eval):
        for pred_idx, pred_note in enumerate(pred_notes_eval):
            if pred_idx in matched_pred_indices_pitch:
                continue
            onset_match = abs(gt_note["start_time"] - pred_note["start_time"]) <= onset_window
            pitch_match = gt_note["pitch"] == pred_note["pitch"]
            if onset_match and pitch_match:
                matched_pred_indices_pitch.add(pred_idx)
                matched_gt_indices_pitch.add(gt_idx)
                break

    n_pitch_matches = len(matched_gt_indices_pitch)

    # count how many also have correct string+fret
    matched_pred_indices_tab = set()
    matched_gt_indices_tab = set()
    for gt_idx, gt_note in enumerate(gt_notes_eval):
        for pred_idx, pred_note in enumerate(pred_notes_eval):
            if pred_idx in matched_pred_indices_tab:
                continue
            onset_match = abs(gt_note["start_time"] - pred_note["start_time"]) <= onset_window
            pitch_match = gt_note["pitch"] == pred_note["pitch"]
            string_match = gt_note["string"] == pred_note["string"]
            fret_match = gt_note["fret"] == pred_note["fret"]
            if onset_match and pitch_match and string_match and fret_match:
                matched_pred_indices_tab.add(pred_idx)
                matched_gt_indices_tab.add(gt_idx)
                break

    n_tab_matches = len(matched_gt_indices_tab)

    # TDR: correct string-fret pairs / correct pitch matches
    tdr = n_tab_matches / n_pitch_matches if n_pitch_matches > 0 else 0.0

    # Tab metrics: precision, recall, F1 
    tp_tab = n_tab_matches
    p_tab = tp_tab / len(pred_notes_eval) if pred_notes_eval else (1.0 if not gt_notes_eval else 0.0)
    r_tab = tp_tab / len(gt_notes_eval) if gt_notes_eval else (1.0 if not pred_notes_eval else 0.0)
    f1_tab = 2 * p_tab * r_tab / (p_tab + r_tab) if (p_tab + r_tab) > 0 else 0.0

    return {"tdr": tdr, "tab_precision": p_tab, "tab_recall": r_tab, "tab_f1": f1_tab}

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
            # MPE metrics (frame-level)
            silence_fret_idx = mt_config.MAX_FRETS + mt_config.FRET_SILENCE_CLASS_OFFSET if hasattr(mt_config, 'FRET_SILENCE_CLASS_OFFSET') else 21
            fret_padding_value = mt_config.FRET_PADDING_VALUE if hasattr(mt_config, 'FRET_PADDING_VALUE') else -100
            gt_frets = notes_to_fret_matrix(
                gt_notes,
                n_frames,
                n_strings,
                mt_config.HOP_LENGTH,
                mt_config.SAMPLE_RATE,
                mt_config.MAX_FRETS,
                silence_fret_idx
            )
            mpe_metrics = calculate_mpe_metrics(pred_frets, gt_frets, silence_fret_idx, fret_padding_value)

            all_scores.append({
                "model": model_name,
                "track": stem,
                **tdr_metrics,
                **onset_metrics,
                **mpe_metrics,
            })
            print(f"{stem} {model_name}: TDR={tdr_metrics['tdr']:.3f}, Tab_F1={tdr_metrics['tab_f1']:.3f}, Tab_P={tdr_metrics['tab_precision']:.3f}, Tab_R={tdr_metrics['tab_recall']:.3f}, Onset_F1={onset_metrics['onset_f1_event']:.3f}, MPE_F1={mpe_metrics['mpe_f1']:.3f}")

    df = pd.DataFrame(all_scores)
    output_dir = os.path.join(output_base, "evaluation_transcription")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "scores.csv"), index=False)
    # Compute and save mean scores for all relevant metrics
    metric_cols = [
        "tdr", "tab_precision", "tab_recall", "tab_f1",
        "onset_precision_event", "onset_recall_event", "onset_f1_event",
        "mpe_precision", "mpe_recall", "mpe_f1"
    ]
    mean_scores = df.groupby("model")[metric_cols].mean()
    print("\n" + "="*20)
    print("Mean Scores (per model)")
    print("="*20)
    print(mean_scores)
    mean_scores.to_csv(os.path.join(output_dir, "mean_scores.csv"))

if __name__ == "__main__":
    main()
