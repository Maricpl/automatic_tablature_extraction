from abc import ABC, abstractmethod

import librosa as lr
import argparse
import torch
import numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
from matplotlib import pyplot as plt
from tayuya import MIDIParser
import os

from music_transcription.python.model.architecture import GuitarTabCRNN
from music_transcription.python.evaluation.tablature_export import (
    _generate_tablature_matrix_slots,
    _format_tablature_matrix_to_text,
)
import pretty_midi
from music_transcription.python.training.note_conversion_utils import frames_to_notes_for_eval

from music_transcription.python.model.utils import load_best_model
import music_transcription.python.config as crnn_config


class TranscriptionModel(ABC):
    def __init__(self, model_name: str, output_dir: str = None):
        self.model_name = model_name
        if output_dir is None:
            output_dir = "data/results/" + model_name + "/"
        self.output_dir = output_dir

    @abstractmethod
    def transcribe(self, input_path: str) -> dict:
        """
        Abstract method to transcribe audio into MIDI.

        :param input_path: Path to the audio file to be separated.
        :return: Path to output MIDI or tabs file.
        """
        pass


class BasicPitch(TranscriptionModel):
    def __init__(self, output_dir: str = None):
        super().__init__(model_name="basic_pitch", output_dir=output_dir)

    def transcribe(self, input_path: str) -> dict:
        model_output, midi_data, note_events = predict(
            input_path, ICASSP_2022_MODEL_PATH
        )

        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_dir_for_stem = os.path.join(self.output_dir)
        os.makedirs(output_dir_for_stem, exist_ok=True)
        output_path = os.path.join(output_dir_for_stem, f"{stem}.mid")

        midi_data.write(output_path)
        print(f"Transcription completed. MIDI data saved to {output_path}")

        return {"midi_path": output_path, "midi_data": midi_data}

    def plot_piano_roll(self, midi_data, start_pitch=24, end_pitch=84, fs=100):
        # Use librosa's specshow function for displaying the piano roll
        fig, ax = plt.subplots(figsize=(10, 4))
        img = lr.display.specshow(
            midi_data.get_piano_roll(fs)[start_pitch:end_pitch],
            hop_length=1,
            sr=fs,
            x_axis="time",
            y_axis="cqt_note",
            fmin=pretty_midi.note_number_to_hz(start_pitch),
            ax=ax,
        )

        return fig


class CRNN(TranscriptionModel):
    def __init__(self, output_dir: str = None, model_path="models/crnn/model_crnn.pth", run_config_path="models/crnn/run_configuration.json"):
        super().__init__(model_name="crnn", output_dir=output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_best_model(
            GuitarTabCRNN,
            model_path,
            run_config_path,
            self.device
        )

    def transcribe(self, input_path: str, onset_threshold: float = 0.46) -> dict:
        audio, sr = lr.load(input_path, sr=crnn_config.SAMPLE_RATE, mono=True)
        cqt = lr.cqt(
            y=audio,
            sr=sr,
            hop_length=crnn_config.HOP_LENGTH,
            fmin=crnn_config.FMIN_CQT,
            n_bins=crnn_config.N_BINS_CQT,
            bins_per_octave=crnn_config.BINS_PER_OCTAVE_CQT,
        )
        log_cqt = lr.amplitude_to_db(np.abs(cqt), ref=np.max)
        features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            onset_logits, fret_logits = self.model(features)

        onset_probs = torch.sigmoid(onset_logits).squeeze(0).cpu()
        fret_indices = torch.argmax(fret_logits, dim=-1).squeeze(0).cpu()

        # Binarize onsets for note extraction
        onset_bin = (onset_probs > onset_threshold).float()

        # Use original repo's note extraction
        notes_list = frames_to_notes_for_eval(
            onset_bin,
            fret_indices,
            crnn_config.HOP_LENGTH,
            crnn_config.SAMPLE_RATE,
            max_fret_value=crnn_config.MAX_FRETS,
            min_note_duration_frames=crnn_config.MIN_NOTE_DURATION_FRAMES,
            open_string_pitches=crnn_config.OPEN_STRING_PITCHES_MIDI
        )

        # Save onsets and frets as .npz
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_dir_for_stem = os.path.join(self.output_dir)
        os.makedirs(output_dir_for_stem, exist_ok=True)
        npz_path = os.path.join(output_dir_for_stem, f"{stem}_onsets_frets.npz")
        np.savez(npz_path, onsets=onset_probs.numpy(), frets=fret_indices.numpy())

        # Create MIDI using pretty_midi
        midi = pretty_midi.PrettyMIDI()
        guitar_program = crnn_config.ACOUSTIC_GUITAR_STEEL_PROGRAM if hasattr(crnn_config, 'ACOUSTIC_GUITAR_STEEL_PROGRAM') else 25
        instrument = pretty_midi.Instrument(program=guitar_program)
        for note in notes_list:
            midi_note = pretty_midi.Note(
                velocity=crnn_config.DEFAULT_MIDI_VELOCITY if hasattr(crnn_config, 'DEFAULT_MIDI_VELOCITY') else 100,
                pitch=note['pitch_midi'],
                start=note['start_time'],
                end=note['end_time']
            )
            instrument.notes.append(midi_note)
        midi.instruments.append(instrument)

        # Generate tablature text
        tab_matrix = _generate_tablature_matrix_slots(
            onset_probs,
            fret_indices,
            onset_probs.shape[0],
            crnn_config.DEFAULT_NUM_STRINGS,
            crnn_config.MAX_FRETS,
            onset_threshold,
        )
        tab_text = _format_tablature_matrix_to_text(
            tab_matrix, crnn_config.DEFAULT_NUM_STRINGS
        )

        tab_path = os.path.join(output_dir_for_stem, f"{stem}_thresh{onset_threshold:.2f}.txt")
        midi_path = os.path.join(output_dir_for_stem, f"{stem}_thresh{onset_threshold:.2f}.mid")

        with open(tab_path, "w") as f:
            f.write(tab_text)
        midi.write(midi_path)

        print(f"Tablature saved to {tab_path}")
        print(f"MIDI saved to {midi_path}")
        print(f"Onsets/frets saved to {npz_path}")
        return {"tabs_path": tab_path, "tabs_content": tab_text, "midi_path": midi_path, "midi_data": midi, "onsets_frets_path": npz_path}

class Tayuya(TranscriptionModel):
    def __init__(self, output_dir: str = None, source_model_name: str = None):
        super().__init__(model_name="tayuya", output_dir=output_dir)
        self.source_model_name = source_model_name

    def transcribe(self, input_path: str) -> dict:
        mid = MIDIParser(input_path, track=1)
        tabs = mid.render_tabs()
        # Extract note events and approximate onsets/frets
        notes_played = mid.notes_played()  # List[Dict] with 'note' and 'time'
        key = mid.get_key()
        from tayuya.tabs import Tabs
        tabs_obj = Tabs(notes=notes_played, key=key)
        notes_with_pos = tabs_obj.generate_notes()  # List of (note, string, fret)
        # Build onsets/frets arrays to match CRNN format: (num_frames, num_strings)
        num_notes = len(notes_with_pos)
        num_strings = 6
        # Use -100 for empty frets (CRNN uses this as FRET_PADDING_VALUE)
        onsets = np.zeros((num_notes, num_strings), dtype=np.float32)
        frets = np.full((num_notes, num_strings), fill_value=-100, dtype=np.int32)
        for idx, (note, string, fret) in enumerate(notes_with_pos):
            s = int(string) - 1  # Tayuya uses 1-based string index
            onsets[idx, s] = 1.0
            frets[idx, s] = fret
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = self.output_dir
        if self.source_model_name:
            output_dir = os.path.join(output_dir, self.source_model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{stem}.txt")
        npz_path = os.path.join(output_dir, f"{stem}_onsets_frets.npz")
        with open(output_path, "w") as f:
            f.write(tabs)
        np.savez(npz_path, onsets=onsets, frets=frets)
        print(f"Tablature saved to {output_path}")
        print(f"Onsets/frets saved to {npz_path}")
        # Optionally, build a notes_list for compatibility
        notes_list = []
        for idx, (note, string, fret) in enumerate(notes_with_pos):
            notes_list.append({
                'start_time': idx,  # No real timing info
                'end_time': idx + 1,
                'string': int(string) - 1,
                'fret': fret,
                'note': note
            })
        return {"tabs_path": output_path, "tabs_content": tabs, "notes_list": notes_list, "onsets_frets_path": npz_path}


class TrascriptionHub:
    _model_mapping = {
        "basic_pitch": BasicPitch,
        "crnn": CRNN,
    }

    def __init__(self, model_name: str, output_dir: str = None):
        if model_name not in self._model_mapping:
            raise ValueError(
                f"Invalid model name '{model_name}'. Available models: {self.get_available_models()}"
            )
        self.model = self._model_mapping[model_name](output_dir=output_dir)
        self.model_name = model_name
        self.midi_to_tabs_model = None
        if model_name != "crnn":
            self.midi_to_tabs_model = Tayuya(source_model_name=self.model_name)

    def transcribe(self, input_path: str, to_tabs: bool = True) -> dict:
        transcription_result = self.model.transcribe(input_path)

        if to_tabs and self.midi_to_tabs_model:
            if "midi_path" in transcription_result:
                tabs_result = self.midi_to_tabs_model.transcribe(
                    transcription_result["midi_path"]
                )
                transcription_result.update(tabs_result)
            else:
                print("Model does not output MIDI, cannot convert to tabs with Tayuya.")

        return transcription_result

    def plot_piano_roll(self, midi_data, start_pitch=24, end_pitch=84, fs=100):
        if isinstance(self.model, BasicPitch):
            return self.model.plot_piano_roll(midi_data, start_pitch, end_pitch, fs)
        else:
            # Generic plot for other models if they output MIDI
            if not hasattr(midi_data, 'get_piano_roll'):
                print("Cannot plot piano roll for this model's output.")
                return None
            fig, ax = plt.subplots(figsize=(10, 4))
            img = lr.display.specshow(
                midi_data.get_piano_roll(fs)[start_pitch:end_pitch],
                hop_length=1,
                sr=fs,
                x_axis="time",
                y_axis="cqt_note",
                fmin=pretty_midi.note_number_to_hz(start_pitch),
                ax=ax,
            )
            return fig

    @classmethod
    def get_available_models(cls):
        """
        Returns a list of available models for transcription.
        """
        return list(cls._model_mapping.keys())


def get_module_args():
    parser = argparse.ArgumentParser(
        description="Automatic Music Transcription",
        epilog="Example usage: python -m tablature_extraction.transcription --model basic_pitch --input_path data/example.wav",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=TrascriptionHub.get_available_models(),
        default="basic_pitch",
        help="Model to use for transcription",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the audio file to be transcribed",
    )
    parser.add_argument(
        "--to_tabs",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Convert transcription output to tabs (if applicable).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_module_args()

    model = TrascriptionHub(model_name=args.model)
    result = model.transcribe(args.input_path, to_tabs=args.to_tabs)

    if "tabs_content" in result:
        print("\n--- Generated Tablature ---")
        print(result["tabs_content"])
