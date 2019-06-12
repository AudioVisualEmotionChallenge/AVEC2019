Baseline scripts for the Cross-cultural Emotion Sub-Challenge (CES); main functions:

baseline_lstm.py: Perform training of a 2-layer LSTM on the features and save predictions.

calc_scores.py: Calculate Concordance Correlation Coefficient (CCC), Pearson's Correlation Coefficient (PCC) and Mean Squared Error (MSE) on the concatenated predictions. Note: Only the CCC is taken into account as the official metric for the challenge.
