# Automatic tablature extraction
Master's thesis at Warsaw Univerity of Technology.
Topic: Music Source Separation and Auomatic Music Transciption using Artificial Intelligence methods.

Topic is devoted into problem of generating guitar tablature from songs recordings. It's related to two Music Information Retrieval fields: Music Source Separation and Automatic Music Trascription.

## REPO structure
experiments.ipynb - contains simple inference tests of Open-Unmix and Hybrid Demucs for source separation and baisc-pitch for music transcription for four example recordings.


# Example run:
```bash
python -m tablature_extraction.pipeline --separation_model open_unmix --transcription_model basic_pitch --audio data/songs/mettalica_10s.wav 
```


# Submodules
There's a need to set the environment variable SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True, because of one of old dependencies.
Apply git patch for submodules:
```bash
cd DTTNetPytorch
git patch ../DTTNetPytorch_imports.patch
cd ..
```



BSRNN model checkpoit for other stem can be downloaded from https://huggingface.co/crlandsc/bsrnn-other/tree/main and should be placed in BandSplitRNN/src/saved_models/other/

# Tests
run with
```bash
python -m pytest tests/
```