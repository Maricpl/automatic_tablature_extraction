# Literature analysis
Markdown for documenting literature analysis progress in structured way.
More detailed description of articles is located in thesis.

Separated into 2 main parts Music Source Separation and Automatic Music Transcription.

## Music Source Separation Models
| Name | Article | Year | Notes | Status code | Status thesis |
|---|---|---|---|---|---|
| **Wave-U-Net** | Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation | 2018 | **Stoller et al.** <br> Time-domain adaptation of U-Net; avoids STFT issues. | Wont ✅| ✅ |
| **Open-Unmix** | Open-Unmix - A Reference Implementation for Music Source Separation | 2019 | **Stöter et al.** <br> Open-source baseline using BiLSTMs; reference for many studies. | ✅ | ✅ |
| **Demucs** | Music Source Separation in the Waveform Domain | 2019 | **Défossez et al.** <br> Waveform-to-waveform U-Net with GLUs; strong bass/drums performance. | Lower version ✅ | ✅ |
| **Conv-TasNet** | Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation | 2019 | **Luo & Mesgarani** <br> Time-domain dilated convolutional network; originally for speech, adapted for music. | Wont ✅| ✅ |
| **Spleeter** | Spleeter: a fast and efficient music source separation tool with pre-trained models | 2020 | **Hennequin et al.** <br> Deezer's industrial tool; efficient U-Net based TensorFlow implementation. | Dependency hell ✅ | ✅ |
| **Hybrid Demucs** | Hybrid Spectrogram and Waveform Source Separation | 2021 | **Défossez** <br> Combines spectral (frequency) and waveform (time) domain branches. | ✅ |✅ |
| **KUIELab-MDX-Net** | KUIELab-MDX-Net: A Two-Stream Neural Network for Music Demixing | 2021 | **Kim et al.** <br> Winning solution for MDX 2021; ensemble of TFC-TDF-U-Net and Demucs. | ✅ another demucs version | partly |
| **TFC-TDF-U-Net** | KUIELab-MDX-Net (TFC-TDF-U-Net) | 2021 | **Choi et al.** <br> Time-Frequency Convolutions and Time-Distributed Fully-connected U-Net. | ✅ another demucs version| partly |
| **Band-split RNN** | Music Source Separation with Band-split RNN | 2022 | **Luo & Yu** <br> Splits spectrogram into subbands processed by RNNs for fine-grained modeling. | ✅ | ✅ |
| **Mel-Band RoFormer** | Mel-Band RoFormer for Music Source Separation | 2023 | **Wang et al.** <br> Transformer with Rotary Embeddings & Mel-band split; SOTA performance. | ✅ | ✅|
| **Band-Split RoPE Transformer** | MUSIC SOURCE SEPARATION WITH BAND-SPLIT ROPE TRANSFORMER | 2023 | **Lu et al.** <br> Replaces RNN in Band-Split model with Transformers; MDX23 winner. | ✅ | ✅|
| **Hybrid Transformer Demucs** | HYBRID TRANSFORMERS FOR MUSIC SOURCE SEPARATION | 2023 | **Rouard et al.** <br> HT Demucs; adds Transformer layers to the Hybrid Demucs architecture. | ✅ | ✅ |
| **SCNet** | SCNet: Sparse Compression Network for Music Source Separation | 2024 | **Tong et al.** <br> Uses sparse compression for efficient and high-quality separation. | ✅ | ✅|
| **Dual-Path TFC-TDF UNet** | | | | ✅ | ✅|
| ** TFC-TDF UNet v3** | | | | ✅ worse metrics on other | ✅|
| **A Stem-Agnostic System** | A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems | 2024 | **Watcharasupat & Lerch** <br> Single decoder architecture for arbitrary stem separation. | ✅ wont |Todo |
| **Multi-Source Diffusion Models** | Multi-Source Diffusion Models for Simultaneous Music Generation and Separation | 2024 | **Mariani et al.** (ICLR) <br> Applies diffusion models to jointly generate and separate sources. | ✅wont|✅|
| **MAJL** | MAJL: A Model-Agnostic Joint Learning Framework for Music Source Separation and Pitch Estimation | 2024 | **Wei et al.** <br> Jointly learns separation and pitch estimation to improve both tasks. | ✅wont| ✅|
| **GASS** | GASS – Generalizing Audio Source Separation with Large-scale Data | 2024 | **Pons et al.** <br> Focuses on generalizing separation across domains using massive datasets. | ✅wont| ✅|
| **MGE-LDM** | MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction | 2025 | **Chae & Lee** <br> Unified latent diffusion framework for generation, imputation, and separation. | ✅wont| ✅|

## Music Source Separation Datasets
| Name | Article | Year | Notes |
|---|---|---|---|
| **MedleyDB** | MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research | 2014 | **Bittner et al.** <br> High-quality multitrack dataset; standard for diverse instrument separation. |
| **Slakh2100** | Cutting Music Source Separation Some Slakh | 2019 | **Manilow et al.** <br> Large synthetic dataset generated from the Lakh MIDI dataset; used for data augmentation. |
| **MoisesDB** | MoisesDB: A Dataset for Source Separation Beyond 4-Stems | 2023 | **Pereira et al.** <br> Focuses on fine-grained stems (e.g., separating guitar, piano, strings). |
| **GuitarDuets** | Classical Guitar Duet Separation Using GuitarDuets | 2024 | **Glytsos et al.** <br> Dataset of real and synthesized classical guitar duets. |
| **MUSDB** |

## Music Source Separation Other
| Name | Article | Year | Notes |
|---|---|---|---|
| **Score-Informed** | Score-Informed Source Separation for Musical Audio Recordings: An Overview | 2013 | **Ewert et al.** <br> Uses musical scores to guide NMF or other separation techniques. |
| **U-Net (Base)** | U-Net: Convolutional Networks for Biomedical Image Segmentation | 2015 | **Ronneberger et al.** <br> The foundational architecture adapted for many modern MSS models (e.g., Demucs, Spleeter). |
| **HPSS / Classical** | Genre Specific Dictionaries for Harmonic/Percussive Source Separation | 2016 | **Laroche et al.** <br> Classical techniques (NMF, HPSS, RPCA); largely superseded by DL but useful for unsupervised tasks. |
| **Singing Voice Separation with Deep U-Net** | Singing Voice Separation with Deep U-Net Convolutional Networks | 2017 | **Jansson et al.** <br> Pioneer U-Net application for audio separation (vocal/accompaniment). |
| **NUSSL** | The Northwestern University Source Separation Library | 2018 | **Manilow et al.** <br> Python library implementing many classic and DL separation algorithms. |
| **Commercial Music** | Towards robust music source separation on loud commercial music | 2022 | **Jeon & Lee** <br> Analyzes the impact of dynamic compression/mastering on separation quality. |
| **Music Separation Enhancement** | MUSIC SEPARATION ENHANCEMENT WITH GENERATIVE MODELING | 2022 | **Schaffer et al.** <br> Uses generative models to refine and improve separation quality. |
| **MERT** | MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training | 2023 | **Li et al.** <br> Self-supervised model for music understanding; useful for downstream feature extraction. |

## Automatic Music Transcription
| Name | Article | Year | Notes |
|---|---|---|---|
| **Probabilistic Subspace** | A PROBABILISTIC SUBSPACE MODEL FOR MULTI-INSTRUMENT POLYPHONIC TRANSCRIPTION | 2010 | **Grindlay & Ellis** <br> Early probabilistic approach to multi-instrument transcription. |
| **NMF Beta-divergence** | Real-time Polyphonic Music Transcription with Non-negative Matrix Factorization... | 2010 | **Dessein et al.** <br> NMF-based approach for real-time transcription. |
| **AMT Overview** | Automatic Music Transcription: Breaking the Glass Ceiling | 2012 | **Benetos et al.** <br> Critical analysis of AMT limitations and future directions. |
| **AMT Overview 2** | Automatic Music Transcription: An Overview | 2013 | **Benetos et al.** <br> Comprehensive review of the field. |
| **RNN LM for AMT** | An RNN-based Music Language Model for Improving Automatic Music Transcription | 2014 | **Sigtia et al.** <br> Integrates a language model to improve note prediction accuracy. |
| **Template Adaptation** | Template Adaptation for Improving Automatic Music Transcription | 2014 | **Benetos et al.** <br> Adapts spectrogram templates to varying timbres. |
| **Drum Transcription** | Automatic Drum Transcription Using Bi-Directional Recurrent Neural Networks | 2016 | **Southall et al.** <br> Specialized RNN architecture for drum transcription. |
| **Onsets and Frames** | Onsets and Frames: Dual-Objective Pia❌Transcription | 2018 | **Hawthorne et al.** <br> Standard model for piano; uses separate heads for onsets and frames. |
| **End-to-end Monophonic** | An End-to-end Framework for Audio-to-Score Music Transcription on Monophonic Excerpts | 2018 | **Román et al.** <br> Direct audio-to-score transcription for monophonic sources. |
| **Evaluating AMT** | Evaluating Automatic Polyphonic Music Transcription | 2018 | **McLeod & Steedman** <br> Defines standard metrics for evaluating polyphonic transcription. |
| **Guitar Tab CNN** | Guitar Tablature Estimation with a Convolutional Neural Network | 2019 | **Wiggins & Kim** <br> CNN-based approach for estimating guitar tablature. |
| **Acoustic + LM** | Blending Acoustic and Language Model Predictions for Automatic Music Transcription | 2019 | **Ycart et al.** <br> Hybrid approach combining acoustic models with symbolic language models. |
| **Holistic AMT** | A Holistic Approach to Polyphonic Music Transcription with Neural Networks | 2019 | **Román et al.** <br> End-to-end neural network approach for polyphonic audio. |
| **Invertible NN** | Towards Interpretable Polyphonic Transcription with Invertible Neural Networks | 2019 | **Kelz & Widmer** <br> Focuses on model interpretability using invertible networks. |
| **AMT Future** | Automatic Music Transcription: Challenges and Future Directions | 2019 | **Benetos et al.** <br> Updated review focusing on the deep learning era. |
| **DL for Transcription** | Music transcription modelling and composition using deep learning | 2020 | **Benetos et al.** <br> Review of deep learning methods in AMT. |
| **MT3** | MT3: Multi-Task Multitrack Music Transcription | 2022 | **Gardner et al.** <br> Transformer (T5) based multi-instrument transcription using tokenized MIDI. |
| **Basic Pitch** | A LIGHTWEIGHT INSTRUMENT-AGNOSTIC MODEL FOR POLYPHONIC NOTE TRANSCRIPTION... | 2022 | **Bittner et al.** <br> Spotify's lightweight, instrument-agnostic model suitable for mobile/web. |
| **FretNet** | FretNet: Continuous-Valued Pitch Contour Streaming for Polyphonic Guitar... | 2022 | **Cwitkowitz et al.** <br> Transcribes tablature and continuous pitch (bends/slides). |
| **Percussive Guitar** | Real-Time Percussive Technique Recognition and Embedding Learning... | 2023 | **Martelloni et al.** <br> Focuses on recognizing percussive guitar techniques. |
| **Seq2Seq Guitar** | Sequence-to-Sequence Network Training Methods for Automatic Guitar Transcription... | 2023 | **Kim et al.** <br> Tokenized output approach for guitar transcription. |
| **MIDI-to-Tab** | MIDI-to-Tab: Guitar Tablature Inference via Masked Language Modeling | 2024 | **Edwards et al.** <br> Uses MLM to infer guitar fingering (tab) from MIDI data. |
| **TART** | TART: A Comprehensive Tool for Technique-Aware Audio-to-Tab Guitar Transcription | 2025 | **Gupta et al.** <br> Framework for technique-aware guitar transcription (bends, slides). |
| **TPMNet** | Multi-task learning-based temporal pattern matching network for guitar tablature... | 2025 | **Kim et al.** <br> Uses temporal pattern matching for robust tablature generation. |

## Automatic Music Transcription Datasets
| Name | Article | Year | Notes |
|---|---|---|---|
| **RWC Music Database** | RWC Music Database: Popular, Classical, and Jazz Music Databases | 2002 | **Goto et al.** <br> Fundamental dataset for MIR; MIDI transcribed by ear. |
| **GuitarSet** | GuitarSet: A Dataset for Guitar Transcription | 2018 | **Xi et al.** <br> Standard acoustic guitar dataset with hexaphonic recordings. |
| **Slakh2100** | Cutting Music Source Separation Some Slakh | 2019 | **Manilow et al.** <br> Synthetic dataset, provides perfectly aligned MIDI and Audio for transcription. |
| **ASAP** | ASAP: a dataset of aligned scores and performances for pia❌transcription | 2020 | **Foscarin et al.** <br> Large dataset of aligned pia❌scores and performances. |
| **EGDB** | Towards Automatic Transcription of Polyphonic Electric Guitar Music | 2022 | **Chen et al.** <br> Electric Guitar Database; focuses on timbre variations and effects. |
| **MoisesDB** | MoisesDB: A Dataset for Source Separation Beyond 4-Stems | 2023 | **Pereira et al.** <br> Contains fine-grained stems useful for multi-instrument transcription. |
| **GAPS** | GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model | 2024 | **Riley et al.** <br> Large-scale classical guitar dataset with rich annotations. |
| **GOAT** | GOAT: A Large Dataset of Paired Guitar Audio Recordings and Tablatures | 2025 | **Loth et al.** <br> Massive dataset of 5.9h of guitar audio paired with tabs (GuitarPro). |