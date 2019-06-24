# Baseline features extraction scripts

## Low-level-descriptors (LLDs)

* extract\_audio\_features.py: Extract acoustic LLDs over time (either eGeMAPS LLDs or MFCCs + delta + acceleration) using openSMILE: http://audeering.com/technology/opensmile/). All audio files in the folder 'audio/' are processed and LLDs are stored in the folder 'audio_features/'.

* extract\_video\_features.py: Extract visual LLDs over time (FAU likelihoods) using openFace: https://github.com/TadasBaltrusaitis/OpenFace/). All video files in the folder 'video/' are processed and LLDs are stored in the folder 'visual_features/'.

* generate\_functionals.py: Compute statistics (mean and standard deviation) of the LLDs using a sliding window.

## Bag-of-Words representations

* generate\_xbow.py: Extract bag-of-audio-words (BoAW) and bag-of-video-words (BoVW) features from the respective LLDs using openXBOW: https://github.com/openXBOW/openXBOW. 
* openXBOW.jar: The openXBOW (Passau Open-Source Crossmodal Bag-of-Words) Toolkit, latest version 1.0.

## Deep learning representations (CNNs)

* Audio/DL\_Audio\_AVEC19.sh: Extract VGG-16 and densenet201 CNNs based audio representations from audio files using 
Deep Spectrum: https://github.com/DeepSpectrum/DeepSpectrum. 
* Video/demo.py: Extract RestNet and VGG-16 VNNs based video representations.
