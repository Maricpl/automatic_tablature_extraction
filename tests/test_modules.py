from tablature_extraction.source_separation import BSRoformer, MelBandRoformer, SCNet, SeparationHub, OpenUnmix, HybridDemucs, HTDemucs, DTTNet, BandSplitRNN, HTDemucsFT, HTDemucsGuitar
from tablature_extraction.transcription import TrascriptionHub, BasicPitch, Tayuya, CRNN
import os
import pytest

@pytest.fixture
def output_dir():
    return "data/tests/"

def test_separation_hub():
    assert SeparationHub.get_available_models() == [
        "open_unmix",
        "hybrid_demucs",
        "ht_demucs",
        "ht_demucs_ft",
        "ht_demucs_guitar",
        "dttnet",
        "bandsplitrnn",
        "bs_roformer",
        "mel_band_roformer",
        "scnet"
    ]

def test_open_unmix(output_dir):
    """Test the OpenUnmix separation model."""
    
    model = OpenUnmix(output_dir + "open_unmix/")
    assert model.model_name == "open_unmix"

    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")
    
def test_hybrid_demucs(output_dir):
    """Test the Hybrid Demucs separation model."""

    model = HybridDemucs(output_dir + "hybrid_demucs/")
    assert model.model_name == "hybrid_demucs"

    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_ht_demucs(output_dir):
    """Test the HT Demucs separation model."""
    
    model = HTDemucs(output_dir + "ht_demucs/")
    assert model.model_name == "ht_demucs"

    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)
    
    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_ht_demucs_ft(output_dir):
    """Test the HT Demucs fine-tuned separation model."""
    
    model = HTDemucsFT(output_dir + "ht_demucs_ft/")
    assert model.model_name == "ht_demucs_ft"

    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)
    
    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_ht_demucs_guitar(output_dir):
    """Test the HT Demucs separation model."""
    
    model = HTDemucsGuitar(output_dir + "ht_demucs_guitar/")
    assert model.model_name == "ht_demucs_guitar"

    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other", "piano", "guitar"] for stem in separated_sources)
    
    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_dttnet(output_dir):
    """Test the DTTNet separation model."""
    
    model = DTTNet(output_dir + "dttnet/")

    assert model.model_name == "dttnet"
    assert model.target == "other"
    assert model.batch_size == 4
    
    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert "other" in separated_sources

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_bandsplitrnn(output_dir):
    """Test the BandSplitRNN separation model."""

    model = BandSplitRNN(output_dir=output_dir + "bandsplitrnn/")
    
    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert "other" in separated_sources

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_bs_roformer(output_dir):
    """Test the BS Roformer separation model."""

    model = BSRoformer(output_dir=output_dir + "bs_roformer/")
    
    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_mel_band_roformer(output_dir):
    """Test the Mel-Band RoFormer separation model."""

    model = MelBandRoformer(output_dir=output_dir + "mel_band_roformer/")
    
    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_scnet(output_dir):
    """Test the SCNet separation model."""

    model = SCNet(output_dir=output_dir + "scnet/")
    
    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_transcription_hub_models():
    assert TrascriptionHub.get_available_models() == [
        "basic_pitch",
        "crnn",
    ]

def test_basic_pitch_transcription(output_dir):
    audio_file = "data/example.wav"
    model = BasicPitch(output_dir=os.path.join(output_dir, "basic_pitch"))
    result = model.transcribe(audio_file)

    assert "midi_path" in result
    assert "midi_data" in result
    assert os.path.isfile(result["midi_path"])
    assert result["midi_path"].endswith(".mid")

def test_tayuya_transcription(output_dir):
    audio_file = "data/example.wav"
    basic_pitch = BasicPitch(output_dir=os.path.join(output_dir, "basic_pitch_for_tayuya"))
    bp_result = basic_pitch.transcribe(audio_file)
    midi_path = bp_result["midi_path"]

    # Test Tayuya alone
    tayuya = Tayuya(output_dir=os.path.join(output_dir, "tayuya"), source_model_name="basic_pitch")
    tayuya_result = tayuya.transcribe(midi_path)

    assert "tabs_path" in tayuya_result
    assert "tabs_content" in tayuya_result
    assert os.path.isfile(tayuya_result["tabs_path"])
    assert tayuya_result["tabs_path"].endswith(".txt")
    
    # Test Tayuya with source_model_name
    tayuya_pipelined = Tayuya(source_model_name="basic_pitch")
    tayuya_pipelined_result = tayuya_pipelined.transcribe(midi_path)
    
    assert "tabs_path" in tayuya_pipelined_result
    assert os.path.isfile(tayuya_pipelined_result["tabs_path"])

    # The path should be data/results/tayuya/basic_pitch/example/example.txt
    assert os.path.join("data", "results", "tayuya", "basic_pitch", "example.txt") == tayuya_pipelined_result["tabs_path"]

def test_crnn_transcription(output_dir):
    audio_file = "data/example.wav"
    model = CRNN(output_dir=os.path.join(output_dir, "crnn"))
    result = model.transcribe(audio_file)

    assert "tabs_path" in result
    assert "tabs_content" in result
    assert os.path.isfile(result["tabs_path"])
    assert result["tabs_path"].endswith(".txt")

    # MIDI checks
    assert "midi_path" in result
    assert "midi_data" in result
    assert os.path.isfile(result["midi_path"])
    assert result["midi_path"].endswith(".mid")
