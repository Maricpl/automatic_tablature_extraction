import argparse
from tablature_extraction.source_separation import SeparationHub
from tablature_extraction.transcription import TrascriptionHub
from matplotlib import pyplot as plt
import os
from tayuya import MIDIParser
import numpy as np
import librosa as lr
import librosa.display


class TablatureGenerationPipeline:
    def __init__(self, separation_model: str, transcription_model: str):
        self.source_separation = SeparationHub(model_name=separation_model)
        self.transcription = TrascriptionHub(model_name=transcription_model)

    def inference(self, audio_path: str):
        # Source Separation
        print(f"Separating sources from {audio_path} using {self.source_separation.model.model_name}...")
        stems = self.source_separation.separate(audio_path)
        guitar_stem = stems["other"]

        # Transcription
        print(f"Transcribing guitar stem from {guitar_stem} using {self.transcription.model.model_name}...")
        midi_data = self.transcription.transcribe(guitar_stem)

        # plt.figure(figsize=(12, 4))
        # self.transcription.plot_piano_roll(midi_data, 24, 84)
        # plt.show()

        stem = "other"
        out_file = self.transcription.model.output_dir + "/" + stem
        print(out_file)
        os.makedirs(out_file, exist_ok=True)
        midi_data.write(out_file +  f'/{stem}.mid')

        mid = MIDIParser(out_file + f'/{stem}.mid', track=1)
        mid.render_tabs()

        # with open(out_file + f'/{stem}.txt', 'w') as f:
        #     for tab in tabs:
        #         f.write(tab + '\n')
        
    def inference_mock(self, audio_path: str):
        # generate spectrogram for audio
        y, sr = lr.load(audio_path, sr=None)
        stft = np.abs(lr.stft(y))
        spectrogram = lr.amplitude_to_db(stft, ref=np.max)
        spectrogram

        fig = plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        plt.tight_layout()

        return fig


def get_module_args():
    parser = argparse.ArgumentParser(
        description="Tablature Generation Pipeline",
        epilog="Example usage: python -m tablature_extraction.pipeline --separation_model open_unmix --transcription_model basic_pitch --audio data/songs/mettalica_10s.wav --output data/results/",
    )
    parser.add_argument(
        "--separation_model",
        type=str,
        choices=SeparationHub.get_available_models(),
        default="open_unmix",
        help="Model to use for source separation",
    )
    parser.add_argument(
        "--transcription_model",
        type=str,
        choices=TrascriptionHub.get_available_models(),
        default="basic_pitch",
        help="Model to use for transcription",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to the audio file to be processed.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_module_args()

    pipeline = TablatureGenerationPipeline(
        separation_model=args.separation_model,
        transcription_model=args.transcription_model,
    )
    pipeline.inference(args.audio)
