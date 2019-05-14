# Baseline features extraction scripts

## Supervised representations

* extract\_audio\_features.py: Extract acoustic features (using openSMILE: http://audeering.com/technology/opensmile/) over time (either eGeMAPS LLDs or MFCCs + delta + acceleration) for all audio files in the folder 'audio/'. Features are stored in the folder 'audio_features/'. The feature script to be extracted needs to be configured in line 11.

* extract\_video\_features.py: Extract visual features (FAU likelihoods, using openFace: https://github.com/TadasBaltrusaitis/OpenFace/) for all video files in the folder 'video/'. Features are stored in the folder 'visual_features/'.

## Semi-supervised representations

* generate\_xbow.py: Extract bag-of-audio-words (BoAW) and bag-of-video-words (BoVW) features (using openXBOW: https://github.com/openXBOW/openXBOW) from the respective low-level descriptors configured in lines 13 and 14. This script uses the tool openXBOW (see below). This script is not required if you have downloaded the features in the folders audio\_features\_xbow/ and visual\_features\_xbow/.

* openXBOW.jar: The openXBOW (Passau Open-Source Crossmodal Bag-of-Words) Toolkit, latest version 1.0.

## Unsupervised representations

Deep Spectrum representation of audio signal can be generated using the DeepSpectrum toolkit: https://github.com/DeepSpectrum/DeepSpectrum. For exact replication of the features, please use the configuration given in: https://github.com/DeepSpectrum/DeepSpectrum#features-for-avec2018-ces.
