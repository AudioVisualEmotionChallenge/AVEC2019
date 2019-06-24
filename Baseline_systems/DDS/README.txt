Baseline Scripts


train_models.py: Trains and validates the model

define_models.py: Defines the model(1-layer 64-d GRU)

create_datasets.py: Handles data loaders

fusion.py: Finds the fusion predictions by taking mean of the regression scores

evaluate_submission.py: Reads submission file to compute score

preprocess.py: Preprocesses all feature files to generate .npy files per data point per feature

run.sh: Runs train_models.py for all feature sets with the required parameters



