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