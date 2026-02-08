# Automatic tablature extraction
Master's thesis at Warsaw Univerity of Technology.
Topic: Music Source Separation and Auomatic Music Transciption using Artificial Intelligence methods.

Topic is devoted into problem of generating guitar tablature from songs recordings. It's related to two Music Information Retrieval fields: Music Source Separation and Automatic Music Trascription.

## REPO structure


# Example run:
```bash
python -m tablature_extraction.pipeline --separation_model open_unmix --transcription_model basic_pitch --audio data/songs/mettalica_10s.wav 
```


# Submodules
There's a need to set the environment variable SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True, because of one of old dependencies.
Apply git patch for submodules:
```bash
cd DTTNetPytorch
git apply ../DTTNetPytorch_imports.patch
cd ..

cd BandSplitRNN
git apply ../BandSplitRNN_imports.patch
cd ..

cd MSS_Trainer
git apply ../MSS_Trainer_imports.patch
cd ..

cd music_transcription
git apply ../music_transcription_imports.patch
cd ..
```
## Downloading checkpoints
DTTNet model checkpoints can be downloaded from https://mega.nz/folder/E4c1QD7Z#OkgM_dEK1tC5MzpqEBuxvQ and should be unzipped to in models/dtt/.

BSRNN model checkpoit for other stem can be downloaded from https://huggingface.co/crlandsc/bsrnn-other/tree/main and should be placed in BandSplitRNN/src/saved_models/other/

Some models were trained by community and shared via https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md. Downloaded checkpoints should be placed in models/ dir.


## Training models
CRNN has been trained using music_transcription repo.
After training model following isntructions, copy best_model.pth to models/dir and name it as model_crnn.pth. Also copy configuration file and palce it in the same directory.

# Unit Tests
run with
```bash
python -m pytest tests/
```

# Evaluation
Evaluation of models were conducted on MUSDB18 dataset - specifically a subset that was derived from MedleyDB (46 songs).
Original files are stores as .mp4 to create wav files follow isntruction in https://github.com/sigsep/sigsep-mus-db. Especially
```bash
musdbconvert data/musdb18 data/musdb18/wav
```
