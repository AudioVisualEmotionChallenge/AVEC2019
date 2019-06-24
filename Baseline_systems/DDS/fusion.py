import os
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from train_models_backup import ccc_score
from sklearn.metrics import mean_squared_error


def fusion_feature_sets(path, val_size):
	fusion_np = np.array([0.0]*val_size)

	files = os.listdir(path)
	files.sort()
	for f in files:
		f_path = os.path.join(path, f)
		file_np = np.load(f_path)
		fusion_np += file_np
	fusion_np = fusion_np / len(files)

	return fusion_np

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_path', default='./data', help='path to dataset')
	parser.add_argument('--results_path', default='./out/predictions/val', help='path to prediction results')
	opt = parser.parse_args()

	val_split = pd.read_csv(os.path.join(opt.dataset_path, 'val_split.csv'), header=0)
	val_labels = val_split['PHQ_Score'].values

	fusion_np = fusion_feature_sets(opt.results_path, len(val_labels))
	fusion_ccc = ccc_score(val_labels/25, fusion_np/25)
	fusion_rmse = sqrt(mean_squared_error(val_labels, fusion_np))
	print("Fusion CCC", fusion_ccc)
	print("Fusion RMSE Score: ", fusion_rmse)

if __name__ == "__main__":
	main()
