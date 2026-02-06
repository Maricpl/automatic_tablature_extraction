from tablature_extraction.source_separation import SeparationHub, OpenUnmix, HybridDemucs, HTDemucs, DTTNet, BandSplitRNN, HTDemucsFT, HTDemucsGuitar
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

    model = SeparationHub(model_name="bandsplitrnn", output_dir=output_dir + "bandsplitrnn/")
    
    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert "other" in separated_sources

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_bs_roformer(output_dir):
    """Test the BS Roformer separation model."""

    model = SeparationHub(model_name="bs_roformer", output_dir=output_dir + "bs_roformer/")
    
    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_mel_band_roformer(output_dir):
    """Test the Mel-Band RoFormer separation model."""

    model = SeparationHub(model_name="mel_band_roformer", output_dir=output_dir + "mel_band_roformer/")
    
    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")

def test_scnet(output_dir):
    """Test the SCNet separation model."""

    model = SeparationHub(model_name="scnet", output_dir=output_dir + "scnet/")
    
    # Run on example.wav
    separated_sources = model.separate("data/example.wav")

    assert all(stem in ["vocals", "drums", "bass", "other"] for stem in separated_sources)

    for stem, path in separated_sources.items():
        assert os.path.isfile(path)
        assert path.endswith(".wav")


