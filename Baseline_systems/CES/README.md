## Baseline scripts for the Cross-cultural Emotion Sub-Challenge (CES)

* baseline\_lstm.py: Perform training of a 2-layer LSTM on the features and save monomodal predictions.

* fuse\_results.py: Perform SVR (late) fusion of the monomodal predictions

* calc\_scores.py: Calculate Concordance Correlation Coefficient (CCC), Pearson's Correlation Coefficient (PCC) and Mean Squared Error (MSE) on the concatenated predictions. Note: Only the CCC is taken into account as the official metric for the challenge.

* evaluate\_submission.py: Script exploited to perform the evaluation of the test results (cannot be reproduced but shared for transparency).
