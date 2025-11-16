import argparse
from tablature_extraction.source_separation import SeparationHub
from tablature_extraction.transcription import TrascriptionHub, TablatureTrascriptionHub
from matplotlib import pyplot as plt
import os
from tayuya import MIDIParser
import numpy as np
import librosa as lr
import librosa.display
from datetime import datetime


class TablatureGenerationPipeline:
    def __init__(self, separation_model: str, transcription_model: str, tablature_transcription_model: str, output_dir: str = None):
        if output_dir is None:
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = "data/results/" + time_str
        os.makedirs(output_dir)
        self.source_separation = SeparationHub(model_name=separation_model, output_dir=output_dir)
        self.transcription = TrascriptionHub(model_name=transcription_model, output_dir=output_dir)
        self.tablature_transcription = TablatureTrascriptionHub(model_name=tablature_transcription_model, output_dir=output_dir)


    def inference(self, audio_path: str, stem="other"):
        # Source Separation
        print(f"Separating sources from {audio_path} using {self.source_separation.model.model_name}...")
        stems = self.source_separation.separate(audio_path)
        guitar_stem = stems["other"]

        # Transcription
        print(f"Transcribing guitar stem from {guitar_stem} using {self.transcription.model.model_name}...")
        midi_data = self.transcription.transcribe(guitar_stem)

        out_file = self.transcription.model.output_dir + "/" + f'/{stem}.mid'
        print(f"Saving MIDI data to {out_file}...")
        # if os.path.exists(out_file):
        #     os.remove(out_file)
        # os.makedirs(out_file, exist_ok=True)
        midi_data.write(out_file)


        # Tablature Transcription
        str_transcription = self.tablature_transcription.transcribe(out_file)

        return guitar_stem, midi_data, str_transcription
    


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
        "--tablature_transcription_model",
        type=str,
        choices=TablatureTrascriptionHub.get_available_models(),
        default="tayuya",
        help="Model to use for tablature transcription",
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
        tablature_transcription_model=args.tablature_transcription_model, 
    )
    pipeline.inference(args.audio)
