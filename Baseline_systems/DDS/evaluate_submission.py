import argparse
import pandas as pd
import numpy as np

from train_models_backup import ccc_score

def evaluation(predictions_file, label_file):

    if predictions_file.endswith('.csv'):
        predictions = pd.read_csv(predictions_file, header=0)['PHQ_Score'].values
    elif predictions_file.endswith('.npy'):
        predictions = np.load(predictions_file)
    labels = pd.read_csv(label_file, header=0)['PHQ_Score'].values

    ccc = ccc_score(labels/25, predictions/25)
    return ccc

def main():

    parser = argparse.ArgumentParser()
    #parser.add_argument('--submission_file_path', default='./submission_file.csv', help='path to dataset file')
    #parser.add_argument('--test_file_path', default='./test_split.csv', help='path to test labels')

    parser.add_argument('--submission_file_path', default='./out/predictions/test/fusion.npy', help='path to dataset file')
    parser.add_argument('--test_file_path', default='__test__csv__file', help='path to test labels')


    opt = parser.parse_args()

    ccc = evaluation(opt.submission_file_path, opt.test_file_path)
    print('CCC Score: ', ccc)

if __name__ == "__main__":
    main()

