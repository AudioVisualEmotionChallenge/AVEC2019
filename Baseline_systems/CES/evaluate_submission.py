import os, sys
import pandas as pd
import numpy as np
from calc_scores import calc_scores


def main():
    if len(sys.argv)>1:
        submission_path = sys.argv[1]
    else:
        submission_path = 'baseline'
    
    dimensions = ['arousal','valence','liking']
    
    gold_standard_path = 'labels_devel/'
    
    num_DE_devel = 14
    num_HU_devel = 14
    
    num_DE = 16
    num_HU = 18
    num_CN = 70
    
    for dim in range(0, len(dimensions)):    
        # Devel
        print('\nDevel German (DE) + Hungarian (HU):')
        evaluate_partition(gold_standard_path, submission_path, dimensions[dim], ['Devel_DE','Devel_HU'], [num_DE_devel,num_HU_devel])
        print('\nDevel German (DE):')
        evaluate_partition(gold_standard_path, submission_path, dimensions[dim], ['Devel_DE'], [num_DE_devel])
        print('\nDevel Hungarian (HU):')
        evaluate_partition(gold_standard_path, submission_path, dimensions[dim], ['Devel_HU'], [num_HU_devel])
        
        # Test
        #print('\nResults German (DE):')
        #evaluate_partition(gold_standard_path, submission_path, dimensions[dim], ['Test_DE'], [num_DE])
        #print('\nResults Hungarian (HU):')
        #evaluate_partition(gold_standard_path, submission_path, dimensions[dim], ['Test_HU'], [num_HU])
        #print('\nResults Chinese (CN):')
        #evaluate_partition(gold_standard_path, submission_path, dimensions[dim], ['Test_CN'], [num_CN])


def evaluate_partition(gold_standard_path, submission_path, dimension, prefixes, num_seqs):
    if submission_path[-1]!='/':    submission_path += '/'
    if gold_standard_path[-1]!='/': gold_standard_path += '/'
    
    # Check if all dimensions (arousal, valence, liking) are in one file (as in the gold standard labels)
    all_in_one = False
    if not os.path.exists(submission_path + dimension):
        all_in_one = True
    
    gold = np.empty(0)
    pred = np.empty(0)
    
    for prefix, num_seq in zip(prefixes, num_seqs):
        for n in range(0, num_seq):
            filename = prefix + '_' + str(n+1).zfill(2) + '.csv'
            # Load gold standard
            gold_n  = pd.read_csv(gold_standard_path + filename, sep=';', usecols=[dimension]).values.flatten()
            gold    = np.concatenate((gold, gold_n))
            # Load predictions and crop them to the length of the gold standard
            if all_in_one:
                pred_n  = pd.read_csv(submission_path + filename, sep=';', usecols=[dimension]).values.flatten()
            else:
                pred_n  = pd.read_csv(submission_path + dimension + '/' + filename, sep=';', usecols=[dimension]).values.flatten()
            if pred_n.shape[0] > gold_n.shape[0]:
                pred_n = pred_n[:gold_n.shape[0]]
            elif pred_n.shape[0] < gold_n.shape[0]:
                missing = gold_n.shape[0]-pred_n.shape[0]
                print('Warning: Missing predictions in file ' + submission_path + filename + ' - repeating last prediction ' + str(missing) + ' times!')
                pred_n = np.concatenate((pred_n, np.repeat(pred_n[-1],missing)))
            pred = np.concatenate((pred, pred_n))
    
    # Compute metrics
    ccc, pcc, rmse = calc_scores(gold,pred)
    print(dimension + ':')
    #print('{:.3f}'.format(ccc), end=' ')
    print('CCC  = {:.3f}'.format(ccc))
    print('PCC  = {:.3f}'.format(pcc))
    print('RMSE = {:.3f}'.format(rmse))
    return ccc, pcc, rmse


if __name__ == '__main__':
    main()
