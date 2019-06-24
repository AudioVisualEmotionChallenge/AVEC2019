

python train_models.py --feature_dim 23 --feature_type eGeMAPS --max_sequence_length 120000 --modality speech --learning_rate 0.0005 --logger_name eGeMAPS.log
python train_models.py --feature_dim 39 --feature_type mfcc --max_sequence_length 120000 --modality speech --learning_rate 0.0001 --logger_name mfcc.log
python train_models.py --feature_dim 49 --feature_type AUpose --max_sequence_length 120000 --modality vision --learning_rate 0.0005 --logger_name AUpose.log

python train_models.py --feature_dim 100 --feature_type BoW_AUpose --max_sequence_length 120000 --modality vision --learning_rate 0.0001 --logger_name BoW_AU.log
python train_models.py --feature_dim 100 --feature_type BoW_eGeMAPS --max_sequence_length 120000 --modality speech --learning_rate 0.0005 --logger_name BoW_eGeMAPS.log
python train_models.py --feature_dim 100 --feature_type BoW_mfcc --max_sequence_length 120000 --modality speech --learning_rate 0.001 --logger_name BoW_mfcc.log


# The frames have been subsampled during preprocessing to fit on memory for the following representations
# Subsampling with a 1/4 ratio for VGG & DS_VGG and 1/2 ratio for ResNet & DS_densenet
# Cropping 20 minutes according to the new frame rate

python train_models.py --feature_dim 4096 --feature_type DS_VGG --max_sequence_length 9000 --modality speech --learning_rate 0.0001 --logger_name DS_VGG.log
python train_models.py --feature_dim 1920 --feature_type DS_densenet --max_sequence_length 18000 --modality speech --learning_rate 0.0005 --logger_name DS_densenet.log
python train_models.py --feature_dim 2048 --feature_type ResNet --max_sequence_length 18000 --modality vision --learning_rate 0.0001 --logger_name ResNet.log
python train_models.py --feature_dim 4096 --feature_type VGG --max_sequence_length 9000 --modality vision --learning_rate 0.001 --logger_name VGG.log

