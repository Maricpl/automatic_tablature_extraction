# Literature analysis
Markdown for documenting literature analysis progress in structured way .
More detailed description of articles is located in thesis.

Separated into 2 main parts Music Source Separation and Automatic Music Transcription

## Music Source Separation
| Name | article | year | status (thesis) | status (code) | notes |
|---|---|---|---|---|---|
Voice Separation — A Local Optimisation Approach | J ¨urgen Kilian, Holger H. Hoos | ISMIR 2002 | | | |
|  Separation of Vocals from Polyphonic Audio Recordings. | Shankar Vembu, Stephan Baumann | ismir 2005 | status (thesis) | status (code) | notes |
| Singing Voice Separation from Monaural Recordings | Yipeng Li, DeLiang Wang  | ISMIR 2006 | status (thesis) | status (code) | notes |
| Separating voices in MIDI | Søren Tjagvad Madsen, Gerhard Widmer | ISMIR 2006 | status (thesis) | status (code) | notes |
| SEMI-SUPERVISED NMF WITH TIME-FREQUENCY ANNOTATIONS FOR SINGLE-CHANNEL SOURCE SEPARATION | Augustin Lefèvre, Francis R. Bach, Cédric Févotte | ismir 2012 | status (thesis) | status (code) | notes |
| MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research. | Rachel M. Bittner, Justin Salamon, Mike Tierney, Matthias Mauch, Chris Cannam, Juan Pablo Bello | ISMIR 2014 | status (thesis) | status (code) | notes |
| Genre Specific Dictionaries for Harmonic/Percussive Source Separation. | Clement Laroche, Hélène Papadopoulos, Matthieu Kowalski, Gaël Richard | ISMIR 2016 | status (thesis) | status (code) | notes |
| Singing Voice Separation with Deep U-Net Convolutional Networks. | Andreas Jansson, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar, Tillman Weyde | ISMIR 2017 | status (thesis) | status (code) | notes |
| Music Source Separation Using Stacked Hourglass Networks | Sungheon Park, Taehoon Kim, Kyogu Lee, Nojun Kwak | ISMIR 2018 | status (thesis) | status (code) | notes |
| The Northwestern University Source Separation Library | Ethan Manilow, Prem Seetharaman, Bryan Pardo | ismir 2018 | status (thesis) | status (code) | notes |
| Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation  | Daniel Stoller, Sebastian Ewert, Simon Dixon | ISMIR 2018 | status (thesis) | status (code) | notes |
| A unified model for zero-shot music source separation, transcription and synthesis  | Liwei Lin, Gus Xia, Qiuqiang Kong, Junyan Jiang | ISMIR 2021 | status (thesis) | status (code) | notes |
| Towards robust music source separation on loud commercial music | Chang-Bin Jeon, Kyogu Lee | ISMIR 2022 | status (thesis) | status (code) | notes |
| MUSIC SEPARATION ENHANCEMENT
WITH GENERATIVE MODELING | Noah Schaffer, Boaz Cogan, Ethan Manilow, Max Morrison, Prem Seetharaman, Bryan Pardo | ISMIR 2022 | status (thesis) | status (code) | IMPROVING SS |
| Music Source Separation With MLP Mixing of Time, Frequency, and Channel | Tomoyasu Nakano, Masataka Goto | ISMIR 2023 | status (thesis) | status (code) | notes |
| MoisesDB: A Dataset for Source Separation Beyond 4-Stems | Igor Pereira, Felipe Araújo, Filip Korzeniowski, Richard Vogl | ISMIR 2023 | status (thesis) | status (code) | notes |
| Classical Guitar Duet Separation Using GuitarDuets - A Dataset of Real and Synthesized Guitar Recordings| Marios Glytsos, Christos Garoufis, Athanasia Zlatintsi, Petros Maragos | ISMIR 2024 | | | 
| Notewise Evaluation for Music Source Separation: A Case Study for Separated Piano Tracks  | Yigitcan Özer, Hans-Ulrich Berendes, Vlora Arifi-Müller, Fabian-Robert Stöter, Meinard Müller | ISMIR 2024 | | |
| A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems  | Karn N. Watcharasupat, Alexander Lerch | ISMIR 2024 | | |
| Mel-RoFormer for Vocal Separation and Vocal Melody Transcription | Ju-Chiang Wang, Wei-Tsung Lu, Jitong Chen | ISMIR 2024 | | |
| Open-Unmix | Open-Unmix - A Reference Implementation for Music Source Separation | | featured | experiments | |
| Spleeter | Spleeter: a fast and efficient music source separation tool with pre-trained models | | featured | | |
| U-Net | U-Net: Convolutional Networks for Biomedical Image Segmentation | | featured | won't | Architecture base for Demucs models family |
| Demucs | Music Source Separation in the Waveform Domain | | featured | todo | |
| Hybrid Demucs | Hybrid Spectrogram and Waveform Source Separation | | featured | experiments | |
| Hybrid Transformer Demucs | HYBRID TRANSFORMERS FOR MUSIC SOURCE SEPARATION | | featured | todo | SOTA | 
| | Score-Informed Source Separation for Musical Audio Recordings: An Overview | | featured | maybe | Improvement in separation with NMF by using music score for time alignment |
| | TOWARDS ROBUST MUSIC SOURCE SEPARATION ON LOUD COMMERCIAL MUSIC | | featured | maybe | Impact of commercial music dynamic compression in domain missmatch |
| Band-split RNN | Music Source Separation with Band-split RNN | | featured | |
| BAND-SPLIT ROPE TRANSFORMER | MUSIC SOURCE SEPARATION WITH BAND-SPLIT ROPE TRANSFORMER | todo |  | |
| Classical source separation techniques RPCA, NMF, HPSS | | | featured | won't | deep learning solutions achieves better metrics
|TDA NET |||||
| GASS | GASS – Generalizing Audio Source Separation with Large-scale Data | 2023 | todo | maybe |
| Source separation Challanges? | todo | | | |

## Automatic Music Transcription
| Name | article | year | status (thesis) | status (code) | notes |
|---|---|---|---|---|---|
 Techniques for Automatic Music Transcription. |  Juan Pablo Bello, Giuliano Monti and Mark Sandler | ISMIR 2000 | | | old low-level basics overwiev |
| Automatic Transcription of Piano Music | Christopher Raphael | ISMIR 2002 | |  | first piano transcription, hidden Markov model |
| An Auditory Model Based Transcriber of Singing Sequences | L. P. Clarisse, J. P. Martens, M. Lesaffre, B. De Baets and M. Leman | ISMIR 2002 |  |  | first vocal transcription, query by humming
Polyphonic Score Retrieval Using Polyphonic Audio Queries: A Harmonic Modeling Approach | Jeremy Pickens , Juan Pablo Bello , Giuliano Monti ,Tim Crawford , Matthew Dovey , Mark Sandler , Don Byrd | ISMIR 2002 | status (thesis) | status (code) | QBH |
RWC Music Database: Popular, Classical, and Jazz Music Databases | Masataka Goto, Hiroki Hashiguchi, Takuichi Nishimura, Ryuichi Oka | ISMIR 2002 | | | MIDI, transcribed by ear
| Pitch Histograms in Audio and Symbolic Music Information Retrieval | George Tzanetakis, Andrey Ermolinskiy, Perry R. Cook | ISMIR 2002 | status (thesis) | status (code) | notes |
| POLYPHONIC MUSIC TRANSCRIPTION BY NON-NEGATIVE SPARSE
CODING OF POWER SPECTRA | Samer A. Abdallah and Mark D. Plumbley | ISMIR 2004 | status (thesis) | status (code) | polyphonic piano music |
| Drum Track Transcription of Polyphonic Music Using Noise Subspace Projection. | Olivier Gillet, Gaël Richard | year | status (thesis) | status (code) | notes |
| A PROBABILISTIC SUBSPACE MODEL FOR MULTI-INSTRUMENT
POLYPHONIC TRANSCRIPTION | Graham Grindlay, Daniel P.W. Ellis | ISMIR 2010 | status (thesis) | status (code) | notes |
| Real-time Polyphonic Music Transcription with Non-negative Matrix Factorization and Beta-divergence | Arnaud Dessein, Arshia Cont, Guillaume Lemaitre | ISMIR 2010 | status (thesis) | status (code) | notes |
| AUTOMATIC MUSIC TRANSCRIPTION: BREAKING THE GLASS CEILING | Emmanouil Benetos, Simon Dixon, Dimitrios Giannoulis, Holger Kirchhoff, and Anssi Klapuri  | ISMIR 2012 | status (thesis) | status (code) | notes |
| An RNN-based Music Language Model for Improving Automatic Music Transcription. | Siddharth Sigtia, Emmanouil Benetos, Srikanth Cherla, Tillman Weyde, Artur S. d’Avila Garcez, Simon Dixon | ISMIR 2014 | status (thesis) | status (code) | notes |
| Template Adaptation for Improving Automatic Music Transcription. | Emmanouil Benetos, Roland Badeau, Tillman Weyde, Gaël Richard | ISMIR 2014 | status (thesis) | status (code) | notes |
| Automatic Drum Transcription Using Bi-Directional Recurrent Neural Networks. | Carl Southall, Ryan Stables, Jason Hockman | ISMIR 2016 | status (thesis) | status (code) | notes |
| GUITARSET: A DATASET FOR GUITAR TRANSCRIPTION | article | year | status (thesis) | status (code) | notes |
| Evaluating Automatic Polyphonic Music Transcription | Andrew McLeod, Mark Steedman | ISMIR 2018 | status (thesis) | status (code) | notes |
| Onsets and Frames: Dual-Objective Piano Transcription | Curtis Hawthorne, Erich Elsen, Jialin Song, Adam Roberts, Ian Simon, Colin Raffel, Jesse Engel, Sageev Oore, Douglas Eck | ismir 2018 | status (thesis) | status (code) | notes |
| An End-to-end Framework for Audio-to-Score Music Transcription on Monophonic Excerpts | Miguel A. Román, Antonio Pertusa, Jorge Calvo-Zaragoza | ISMIR 2018 | status (thesis) | status (code) | notes |
| Guitar Tablature Estimation with a Convolutional Neural Network | Andrew Wiggins, Youngmoo Kim | ISMIR 2019 | status (thesis) | status (code) | notes |
| Blending Acoustic and Language Model Predictions for Automatic Music Transcription | Adrien Ycart, Andrew McLeod, Emmanouil Benetos, Kazuyoshi Yoshii | ISMIR 2019 | status (thesis) | status (code) | notes |
| A Holistic Approach to Polyphonic Music Transcription with Neural Networks | Miguel Roman, Antonio Pertusa, Jorge Calvo-Zaragoza | ismir 2019 | status (thesis) | status (code) | notes |
| Towards Interpretable Polyphonic Transcription with Invertible Neural Networks | Rainer Kelz, Gerhard Widmer | ISMIR 2019 | status (thesis) | status (code) | notes |
| ASAP: a dataset of aligned scores and performances for piano transcription  | Francesco Foscarin, Andrew McLeod, Philippe Rigaux, Florent Jacquemard, Masahiko Sakai | ISMIR 2020 | status (thesis) | status (code) | notes |
| Sequence-to-Sequence Piano Transcription with Transformers | Curtis Hawthorne, Ian Simon, Rigel Swavely, Ethan Manilow, Jesse Engel | ISMIR 2021 | status (thesis) | status (code) | notes |
| Sequence-to-Sequence Network Training Methods for Automatic Guitar Transcription With Tokenized Outputs | Sehun Kim, Kazuya Takeda, Tomoki Toda | ISMIR 2023 | status (thesis) | status (code) | notes |
| Real-Time Percussive Technique Recognition and Embedding Learning for the Acoustic Guitar | Andrea Martelloni, Andrew P. McPherson, Mathieu Barthet | ismir 2023 | status (thesis) | status (code) | notes |
| MIDI-to-Tab: Guitar Tablature Inference via Masked Language Modeling  | Andrew C. Edwards, Xavier Riley, Pedro Pereira Sarmento, Simon Dixon | ISMIR 2024 | status (thesis) | status (code) | MIDI TO TAB |
| GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model  | Xavier Riley, Zixun Guo, Andrew C. Edwards, Simon Dixon | ISMIR 2024 | status (thesis) | status (code) | notes |
| Name | article | year | status (thesis) | status (code) | notes |
| Notes and Multipitch (Basic Pitch)| A LIGHTWEIGHT INSTRUMENT-AGNOSTIC MODEL FOR POLYPHONIC NOTE TRANSCRIPTION AND MULTIPITCH ESTIMATION | | featured | experiments | Spotify lightweight model for Automatic Music Transcription |
| | Automatic Music Transcription: An Overview | | featured | | 
| | Automatic Music Transcription: Challenges and Future Directions | | featured | |
| | Music transcription modelling and composition using deep learning | | featured | |
| MT3 | todo | | | |



Search in ICAASP, AAAI. etc.