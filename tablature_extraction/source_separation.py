from abc import ABC, abstractmethod
from openunmix import predict
import torch
import librosa as lr
import os
import numpy as np
from scipy.io.wavfile import write
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
import torchaudio
import argparse 
from matplotlib import pyplot as plt
#from query_bandit.train import inference_byoq


class SeparationModel(ABC):
    def __init__(self, model_name: str, output_dir: str = None):
        self.model_name = model_name
        if output_dir is None:
            output_dir = "data/results/" + model_name + "/"
        self.output_dir = output_dir

    @abstractmethod
    def separate(self, audio_file: str) -> dict:
        """
        Abstract method to separate audio into different stems.

        :param audio_file: Path to the audio file to be separated.
        :return: Dictionary containing separated sources.
        """
        pass

    def plot_spectrogram(self, audio_file: str):
        """
        Plot the spectrogram of the audio file.

        :param audio_file: Path to the audio file to be plotted.
        """
        y, sr = lr.load(audio_file, sr=None)
        stft = np.abs(lr.stft(y))
        spectrogram = lr.amplitude_to_db(stft, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4))
        img = lr.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="log", ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Spectrogram")
        #fig.tight_layout()

        return fig
    
    # @abc.abstractmethod
    # def load_model(self):
    #     """
    #     Abstract method to load the separation model.
    #     """
    #     pass


class OpenUnmix(SeparationModel):
    def __init__(self, output_dir: str = None):
        super().__init__(model_name="open_unmix", output_dir=output_dir)

    def separate(self, audio_file: str) -> dict:
        """
        Separate the audio file into different sources using OpenUnmix.

        :param audio_file: Path to the audio file to be separated.
        :return: Dictionary containing separated sources.
        """

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        y, sr = lr.load(audio_file, sr=None)

        estimates = predict.separate(torch.as_tensor(y).float(), rate=sr, device=device)
        out_path = self.output_dir + audio_file.split("/")[-1].split(".")[0]
        os.makedirs(out_path, exist_ok=True)

        result = {}
        for target, estimate in estimates.items():
            # print(target)
            audio = estimate.detach().cpu().numpy()[0][0]
            audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
            # display(Audio(audio, rate=sr))
            write(f"{out_path}/{target}.wav", sr, audio)
            print(f"Saved {target} to {out_path}/{target}.wav")
            result[target] = f"{out_path}/{target}.wav"
        
        return result




class HybridDemucs(SeparationModel):
    def __init__(self, output_dir: str = None):
        super().__init__(model_name="hybrid_demucs", output_dir=output_dir)

    def separate(self, audio_file: str) -> dict:
        """
        Separate the audio file into different sources using Hybrid Demucs.

        :param audio_file: Path to the audio file to be separated.
        :return: Dictionary containing separated sources.
        """

        def separate_sources(
            model,
            mix,
            segment=10.0,
            overlap=0.1,
            device=None,
            sample_rate=44100,
        ):
            """
            Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

            Args:
                segment (int): segment length in seconds
                device (torch.device, str, or None): if provided, device on which to
                    execute the computation, otherwise `mix.device` is assumed.
                    When `device` is different from `mix.device`, only local computations will
                    be on `device`, while the entire tracks will be stored on `mix.device`.
            """
            if device is None:
                device = mix.device
            else:
                device = torch.device(device)

            batch, channels, length = mix.shape

            chunk_len = int(sample_rate * segment * (1 + overlap))
            start = 0
            end = chunk_len
            overlap_frames = overlap * sample_rate
            fade = Fade(
                fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear"
            )

            final = torch.zeros(
                batch, len(model.sources), channels, length, device=device
            )

            while start < length - overlap_frames:
                chunk = mix[:, :, start:end]
                with torch.no_grad():
                    out = model.forward(chunk)
                out = fade(out)
                final[:, :, :, start:end] += out
                if start == 0:
                    fade.fade_in_len = int(overlap_frames)
                    start += int(chunk_len - overlap_frames)
                else:
                    start += chunk_len
                end += chunk_len
                if end >= length:
                    fade.fade_out_len = 0
            return final

        bundle = HDEMUCS_HIGH_MUSDB_PLUS

        model = bundle.get_model()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.to(device)

        waveform, sample_rate = torchaudio.load(audio_file)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # print(song)
        # display(Audio(waveform, rate=sample_rate))

        waveform = waveform.to(device)
        mixture = waveform

        segment: int = 10
        overlap = 0.1

        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()  # normalization

        sources = separate_sources(
            model,
            waveform[None],
            device=device,
            segment=segment,
            overlap=overlap,
        )[0]
        sources = sources * ref.std() + ref.mean()

        sources_list = model.sources
        sources = list(sources)

        audios = dict(zip(sources_list, sources))

        result = {}

        out_path = self.output_dir + audio_file.split("/")[-1].split(".")[0]
        os.makedirs(out_path, exist_ok=True)
        for source, audio in audios.items():
            # print(source)
            audio = audio.detach().cpu().numpy()[0]
            audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
            # display(Audio(audio, rate=sample_rate))
            write(f"{out_path}/{source}.wav", sample_rate, audio)
            print(f"Saved {source} to {out_path}/{source}.wav")
            result[source] = f"{out_path}/{source}.wav"

        return result

# class Banquet(SeparationModel):
#     def __init__(self):
#         super().__init__(model_name="banquet")

#     def separate(self, audio_file: str) -> dict:
#         """
#         Separate the audio file into different sources using Banquet.

#         :param audio_file: Path to the audio file to be separated.
#         :return: Dictionary containing separated sources.
#         """
#         inference_byoq()

#         return 0


class SeparationHub(SeparationModel):
    # Class-level mapping of available models
    _model_mapping = {
        "open_unmix": OpenUnmix,
        "hybrid_demucs": HybridDemucs,
        # "banquet": Banquet,
    }

    def __init__(self, model_name: str, output_dir: str = None):
        super().__init__(model_name="separation-hub", output_dir=output_dir)
        if model_name not in self._model_mapping:
            raise ValueError(
                f"Invalid model name '{model_name}'. Available models: {self.get_available_models()}"
            )
        self.model = self._model_mapping[model_name]()

    def separate(self, audio_file: str) -> dict:
        """
        Separate the audio file into different sources using Separation Hub.

        :param audio_file: Path to the audio file to be separated.
        :return: Dictionary containing separated sources.
        """
        return self.model.separate(audio_file)

    @classmethod
    def get_available_models(cls):
        """
        Get the mapping of available models for source separation.

        :return: Dictionary of model names and their corresponding classes.
        """
        return list(cls._model_mapping.keys())


def get_module_args():
    parser = argparse.ArgumentParser(
        description="Audio Source Separation",
        epilog="Example usage: python -m tablature_extraction.source_separation --model open_unmix --audio_file data/songs/mettalica_10s.wav",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=SeparationHub.get_available_models(),
        default="open_unmix",
        help="Model to use for source separation",
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

    model = SeparationHub(model_name=args.model)
    model.separate(args.audio_file)
