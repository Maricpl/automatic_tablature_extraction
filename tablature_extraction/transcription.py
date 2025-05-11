from abc import ABC, abstractmethod

import librosa as lr
import argparse
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
from matplotlib import pyplot as plt
from tayuya import MIDIParser
import os


class TranscriptionModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.output_dir = "data/results/" + model_name

    @abstractmethod
    def transcribe(self, audio_file: str) -> dict:
        """
        Abstract method to transcribe audio into MIDI.

        :param audio_file: Path to the audio file to be separated.
        :return: Dictionary containing separated sources.
        """
        pass


class BasicPitch(TranscriptionModel):
    def __init__(self):
        super().__init__(model_name="basic_pitch")

    def transcribe(self, audio_file: str):
        model_output, midi_data, note_events = predict(
            audio_file, ICASSP_2022_MODEL_PATH
        )

        return midi_data

    def plot_piano_roll(self, midi_data, start_pitch=24, end_pitch=84, fs=100):
        # Use librosa's specshow function for displaying the piano roll
        lr.display.specshow(
            midi_data.get_piano_roll(fs)[start_pitch:end_pitch],
            hop_length=1,
            sr=fs,
            x_axis="time",
            y_axis="cqt_note",
            fmin=pretty_midi.note_number_to_hz(start_pitch),
        )

class TrascriptionHub(TranscriptionModel):
    _model_mapping = {
            "basic_pitch": BasicPitch,
        }
    
    def __init__(self, model_name: str):
        super().__init__(model_name="transcription_hub")
        if model_name not in self._model_mapping:
            raise ValueError(
                f"Invalid model name '{model_name}'. Available models: {self.get_available_models()}"
            )
        self.model = self._model_mapping[model_name]()
        #TODO : Add more models to the mapping - MT3

    def transcribe(self, audio_file: str):
        return self.model.transcribe(audio_file)

    def plot_piano_roll(self, midi_data, start_pitch=24, end_pitch=84, fs=100):
       self.model.plot_piano_roll(midi_data, start_pitch, end_pitch, fs)

    @classmethod
    def get_available_models(cls):
        """
        Returns a list of available models for transcription.
        """
        return list(cls._model_mapping.keys())


def get_module_args():
    parser = argparse.ArgumentParser(
        description="Automatic Music Trabscription",
        epilog="Example usage: python -m tablature_extraction.transcription --model basic_pitch --audio_file data/results/open_unmix/mettalica_10s/other.wav",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=TrascriptionHub.get_available_models(),
        default="basic_pitch",
        help="Model to use for transcription",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to the audio file to be separated",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_module_args()

    model = TrascriptionHub(model_name=args.model)
    midi_data = model.transcribe(args.audio_file)

    plt.figure(figsize=(12, 4))
    model.plot_piano_roll(midi_data, 24, 84)
    plt.show()

    stem = args.audio_file.split("/")[-1].split(".")[0]
    out_file = model.output_dir + "/" + stem
    print(out_file)
    os.makedirs(out_file, exist_ok=True)
    midi_data.write(out_file +  f'/{stem}.mid')

    mid = MIDIParser(out_file + f'/{stem}.mid', track=1)
    print(mid.render_tabs())
