from abc import ABC, abstractmethod
from openunmix import predict
import torch
import librosa as lr
import os
import numpy as np
from scipy.io.wavfile import write
import torchaudio
import argparse 
from matplotlib import pyplot as plt
import demucs.api as demucs_api
from pathlib import Path
import soundfile as sf
from omegaconf import OmegaConf

from BandSplitRNN.src.separator import Separator

from DTTNetPytorch.src.dp_tdf.dp_tdf_net import DPTDFNet
from DTTNetPytorch.src.evaluation.separate import separate_with_ckpt_TDF

from MSS_Training.utils.settings import get_model_from_config, load_config
from MSS_Training.utils.model_utils import load_start_checkpoint, demix, apply_tta
from MSS_Training.utils.audio_utils import normalize_audio, denormalize_audio



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

    def _preprocess_audio(self, y: np.ndarray, sr: int, target_sr: int, target_channels: int):
        if sr != target_sr:
            y = lr.resample(y, orig_sr=sr, target_sr=target_sr)
        
        if y.ndim == 1:
            y = np.stack([y, y], axis=0)

        if target_channels == 1 and y.shape[0] == 2:
            y = lr.to_mono(y)
        
        if target_channels == 2 and y.shape[0] == 1:
            y = np.concatenate([y, y], axis=0)

        return torch.as_tensor(y).float()

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

        y, sr = lr.load(audio_file, sr=None, mono=False)
        y = self._preprocess_audio(y, sr, 44100, 2)

        estimates = predict.separate(y, rate=44100, device=device)
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
    def __init__(self, output_dir: str = None, model_name: str = "hybrid_demucs", demucs_model: str = "hdemucs_mmi"):
        super().__init__(model_name=model_name, output_dir=output_dir)
        self.separator = demucs_api.Separator(model = demucs_model)

    def separate(self, audio_file: str) -> dict:
        """
        Separate the audio file into different sources using Hybrid Demucs.

        :param audio_file: Path to the audio file to be separated.
        :return: Dictionary containing separated sources.
        """

        origin, separated = self.separator.separate_audio_file(audio_file)

        result = {}
        out_path = self.output_dir + audio_file.split("/")[-1].split(".")[0]
        os.makedirs(out_path, exist_ok=True)

        for stem, audio in separated.items():
            demucs_api.save_audio(audio, f"{out_path}/{stem}.wav", samplerate=self.separator.samplerate)

        result[stem] = f"{out_path}/{stem}.wav"

        return result
    
class HTDemucs(HybridDemucs):
    def __init__(self, output_dir: str = None):
        super().__init__(output_dir=output_dir, model_name="ht_demucs", demucs_model="htdemucs")

class HTDemucsFT(HybridDemucs):
    def __init__(self, output_dir: str = None):
        super().__init__(output_dir=output_dir, model_name="ht_demucs_ft", demucs_model="htdemucs_ft")

class HTDemucsGuitar(HybridDemucs):
    def __init__(self, output_dir: str = None):
        super().__init__(output_dir=output_dir, model_name="ht_demucs_guitar", demucs_model="htdemucs_6s")


class DTTNet(SeparationModel):
    """Wrapper for DTTNet (Dual-Path TFC-TDF UNet) source separation model."""
    
    def __init__(self, output_dir: str = None, ckpt_path: str = "models/dtt/otherg32_ep3605.ckpt", target: str = "other", batch_size: int = 4):
        """
        Initialize DTTNet wrapper.
        
        :param output_dir: Directory to save separated sources
        :param ckpt_path: Path to DTTNet checkpoint (.ckpt file)
        :param target: Target stem to extract ("vocals", "drums", "bass", or "other")
        :param batch_size: Batch size for inference
        """
        super().__init__(model_name="dttnet", output_dir=output_dir)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ckpt_path = ckpt_path
        self.target = target
        self.batch_size = batch_size
        
        # Create model with standard DTTNet hyperparameters
        self.model = DPTDFNet(
            dim_f=2048,
            dim_t=256,
            n_fft=6144,
            hop_length=1024,
            overlap=3072,
            audio_ch=2,
            block_type="TFC_TDF_Res2",
            num_blocks=5,
            l=3,
            g=32,
            k=3,
            bn=8,
            bias=False,
            bn_norm="BN",
            bandsequence={
                "rnn_type": "LSTM",
                "bidirectional": True,
                "num_layers": 4,
                "n_heads": 2,
            },
            target_name = 'other',
            lr = 0.0001,
            optimizer = "adamW",
        )
        
        self.separate_with_ckpt = separate_with_ckpt_TDF
    
    def separate(self, audio_file: str) -> dict:
        """
        Separate audio file using DTTNet.
        
        :param audio_file: Path to the audio file to be separated
        :return: Dictionary containing path to separated target stem
        """
        if self.ckpt_path is None:
            raise ValueError(
                "ckpt_path must be provided to DTTNet. "
                "Download a pretrained checkpoint and pass its path."
            )
        
        # Load audio
        y, sr = lr.load(audio_file, sr=None, mono=False)
        mix = self._preprocess_audio(y, sr, 44100, 2).cpu().numpy()
        
        # Separate
        target_wav = self.separate_with_ckpt(
            batch_size=self.batch_size,
            model=self.model,
            ckpt_path=Path(self.ckpt_path),
            mix=mix,
            device=self.device,
            double_chunk=False,
            overlap_add=None
        )
        
        # Save output
        out_path = os.path.join(
            self.output_dir,
            os.path.splitext(os.path.basename(audio_file))[0]
        )
        os.makedirs(out_path, exist_ok=True)
        
        output_file = os.path.join(out_path, f"{self.target}.wav")
        sf.write(output_file, target_wav.T, 44100)
        
        print(f"Saved {self.target} to {output_file}")
        
        return {self.target: output_file}


class BandSplitRNN(SeparationModel):
    """Wrapper for BandSplitRNN source separation model."""
    
    def __init__(self, output_dir: str = None, ckpt_path: str = "BandSplitRNN/src/saved_models/other/other.ckpt", target: str = "other"):
        """
        Initialize BandSplitRNN wrapper.
        
        :param output_dir: Directory to save separated sources
        :param ckpt_path: Path to BandSplitRNN checkpoint (.pt file)
        :param target: Target stem to extract ("vocals", "drums", "bass", or "other")
        """
        super().__init__(model_name="bandsplitrnn", output_dir=output_dir)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ckpt_path = ckpt_path
        self.target = target
        
        # Load config
        self.cfg_path = Path(f"BandSplitRNN/src/saved_models/{self.target}/hparams.yaml")
        self.cfg = OmegaConf.load(self.cfg_path)
        
        # Create model
        self.model = Separator(self.cfg, self.ckpt_path)
        self.model.to(self.device)
        self.model.eval()

    def separate(self, audio_file: str) -> dict:
        """
        Separate audio file using BandSplitRNN.
        
        :param audio_file: Path to the audio file to be separated
        :return: Dictionary containing path to separated target stem
        """
        # Load audio
        y, sr = lr.load(audio_file, sr=None, mono=False)
        mix = self._preprocess_audio(y, sr, 44100, 2)
        mix = mix.to(self.device)

        # Separate
        target_wav = self.model(mix)
        target_wav = target_wav.cpu().numpy()
        
        # Save output
        out_path = os.path.join(
            self.output_dir,
            os.path.splitext(os.path.basename(audio_file))[0]
        )
        os.makedirs(out_path, exist_ok=True)
        
        output_file = os.path.join(out_path, f"{self.target}.wav")
        sf.write(output_file, target_wav.T, 44100)
        
        print(f"Saved {self.target} to {output_file}")
        
        return {self.target: output_file}


class MSSModel(SeparationModel):
    def __init__(self, model_name: str, model_type: str, output_dir: str = None, ckpt_path: str = None, config_path: str = None):
        super().__init__(model_name=model_name, output_dir=output_dir)
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.model_type = model_type

        self.model, self.config = get_model_from_config(self.model_type, self.config_path)
        
        class Args:
            start_check_point = self.ckpt_path
            lora_checkpoint_loralib = None
            model_type = self.model_type

        checkpoint = torch.load(self.ckpt_path, map_location='cuda')
        load_start_checkpoint(Args(), self.model, checkpoint, type_='inference')

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = 'cuda'
        
        self.model = self.model.to(self.device)
        self.model.eval()

    def separate(self, audio_file: str) -> dict:
        sample_rate = getattr(self.config.audio, "sample_rate", 44100)
        
        mix, sr = lr.load(audio_file, sr=sample_rate, mono=False)

        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if "num_channels" in self.config.audio:
                if self.config.audio["num_channels"] == 2:
                    mix = np.concatenate([mix, mix], axis=0)

        norm_params = None
        if "normalize" in self.config.inference:
            if self.config.inference["normalize"] is True:
                mix, norm_params = normalize_audio(mix)

        waveforms = demix(
            self.config,
            self.model,
            mix,
            self.device,
            model_type=self.model_type,
        )

        waveforms = apply_tta(
            self.config,
            self.model,
            mix,
            waveforms,
            self.device,
            self.model_type
        )

        instruments = self.config.training.instruments
        result = {}
        out_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(audio_file))[0])
        os.makedirs(out_path, exist_ok=True)

        for instr in instruments:
            estimates = waveforms[instr]
            if "normalize" in self.config.inference:
                if self.config.inference["normalize"] is True:
                    estimates = denormalize_audio(estimates, norm_params)
            
            output_file = os.path.join(out_path, f"{instr}.wav")
            sf.write(output_file, estimates.T, sample_rate)
            result[instr] = output_file
        
        return result

class BSRoformer(MSSModel):
    def __init__(self, output_dir: str = None, ckpt_path: str = "models/model_bs_roformer_ep_17_sdr_9.6568.ckpt", config_path: str = "configs/bs_roformer_config.yml"):
        super().__init__(
            model_name="bs_roformer",
            model_type="bs_roformer",
            output_dir=output_dir,
            ckpt_path=ckpt_path,
            config_path=config_path
        )

class MelBandRoformer(MSSModel):
    def __init__(self, output_dir: str = None, ckpt_path: str = "models/model_mel_band_roformer_ep_1_sdr_8.2175.ckpt", config_path: str = "configs/mel_band_roformer.yml"):
        super().__init__(
            model_name="mel_band_roformer",
            model_type="mel_band_roformer",
            output_dir=output_dir,
            ckpt_path=ckpt_path,
            config_path=config_path
        )

class SCNet(MSSModel):
    def __init__(self, output_dir: str = None, ckpt_path: str = "models/model_scnet_masked_ep_111_sdr_9.8286.ckpt", config_path: str = "configs/scnet_config.yml"):
        super().__init__(
            model_name="scnet",
            model_type="scnet_masked",
            output_dir=output_dir,
            ckpt_path=ckpt_path,
            config_path=config_path
        )


class SeparationHub(SeparationModel):
    # Class-level mapping of available models
    _model_mapping = {
        "open_unmix": OpenUnmix,
        "hybrid_demucs": HybridDemucs,
        "ht_demucs": HTDemucs,
        "ht_demucs_ft": HTDemucsFT,
        "ht_demucs_guitar": HTDemucsGuitar,
        "dttnet": DTTNet,
        "bandsplitrnn": BandSplitRNN,
        "bs_roformer": BSRoformer,
        "mel_band_roformer": MelBandRoformer,
        "scnet": SCNet
    }


    def __init__(self, model_name: str, output_dir: str = None):
        super().__init__(model_name="separation-hub", output_dir=output_dir)
        if model_name not in self._model_mapping:
            raise ValueError(
                f"Invalid model name '{model_name}'. Available models: {self.get_available_models()}"
            )
        self.model = self._model_mapping[model_name](output_dir=output_dir)

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
