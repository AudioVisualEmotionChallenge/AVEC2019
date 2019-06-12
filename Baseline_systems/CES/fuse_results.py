import os, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from calc_scores import calc_scores

def main():
    predictions = ['baseline_predictions/predictions_egemaps-func_STD',
                   'baseline_predictions/predictions_egemaps-xbow',
                   'baseline_predictions/predictions_mfcc-func_STD',
                   'baseline_predictions/predictions_mfcc-xbow',
                   'baseline_predictions/predictions_deepspectrum',
                   'baseline_predictions/predictions_visual-func',
                   'baseline_predictions/predictions_visual-xbow',
                   'baseline_predictions/predictions_visual-resnet',
                   'baseline_predictions/predictions_visual-vgg']
    num_steps = 1768
    dimensions = ['arousal','valence','liking']
    
    gold_standard_path = 'labels_devel/'
    
    prefixes_devel = ['Devel_DE','Devel_HU']
    num_devel      = [14,14]
    
    prefixes = ['Devel_DE','Devel_HU','Test_DE','Test_HU','Test_CN']
    num_sub  = [14,14,16,18,70]
    
    # Get all devel predictions and gold standard
    pred, gold = read_gold_predictions(predictions[0], prefixes_devel, num_devel, gold_standard_path, dimensions)
    num_samples = gold.shape[0]
    pred_models = np.empty((num_samples,len(dimensions),len(predictions)))
    for ind in range(len(predictions)):
        pred, gold = read_gold_predictions(predictions[ind], prefixes_devel, num_devel, gold_standard_path, dimensions)
        pred_models[:,:,ind] = pred
    
    ## Train a linear regression model for arousal/valence/liking
    #reg_models = []
    #for dim in range(len(dimensions)):
    #    reg_models.append( LinearRegression().fit(pred_models[:,dim,:], gold[:,dim]) )  # HERE
    
    # Train an SVR model for arousal/valence/liking, optimise context width and complexity on the development set
    contexts = [11,13,15,17,19]
    complexities = [0.001,0.01,0.1,1.0]
    ccc_best       = np.zeros(3,)
    context_opt    = np.zeros(3,)
    complexity_opt = np.zeros(3,)
    for context in contexts:
        pred_models_context = contextualise(pred_models, context_past=context, context_future=context)
        for complexity in complexities:  
            svr_models = []
            for dim in range(len(dimensions)):
                svr_models.append( LinearSVR(C=complexity, random_state=0).fit(pred_models_context.reshape(pred_models_context.shape[0],-1), gold[:,dim]) )  # Use all dimensions to train the model for each dimension
                scores = calc_scores( gold[:,dim], svr_models[-1].predict( pred_models_context.reshape(pred_models_context.shape[0],-1) ) )
                if scores[0]>ccc_best[dim]:
                    ccc_best[dim] = scores[0]
                    context_opt[dim] = context
                    complexity_opt[dim] = complexity
                    print(str(context) + ' ' + str(complexity) + ' ' + str(dim) + ' ' + str(scores[0]))
    svr_models = []
    for dim in range(len(dimensions)):
        pred_models_context = contextualise(pred_models, context_past=context_opt[dim], context_future=context_opt[dim])
        svr_models.append( LinearSVR(C=complexity_opt[dim], random_state=0).fit(pred_models_context.reshape(pred_models_context.shape[0],-1), gold[:,dim]) )  # Use all dimensions to train the model for each dimension    
    
    ## Fusion
    ## Baseline: Mean prediction ( np.mean(pred_models, axis=2) )
    #outfolder = 'predictions_mean'
    #for prefix, num in zip(prefixes,num_sub):
    #    for n in range(num):
    #        pred_s = np.empty((num_steps,len(dimensions),len(predictions)))
    #        for ind in range(len(predictions)):
    #            pred_n = read_predictions(predictions[ind], prefix, n, dimensions)
    #            pred_s[:,:,ind] = pred_n
    #        pred_new = np.mean(pred_s, axis=2)  # HERE
    #        write_prediction_file(outfolder, pred_new, dimensions, prefix, n)
    ## Linear regression
    #outfolder = 'predictions_linreg'
    #for prefix, num in zip(prefixes,num_sub):
    #    for n in range(num):
    #        pred_s = np.empty((num_steps,len(dimensions),len(predictions)))
    #        for ind in range(len(predictions)):
    #            pred_n = read_predictions(predictions[ind], prefix, n, dimensions)
    #            pred_s[:,:,ind] = pred_n
    #        pred_new = np.empty((num_steps,len(dimensions)))
    #        for dim in range(len(dimensions)):
    #            pred_new[:,dim] = reg_models[dim].predict(pred_s[:,dim,:])
    #        write_prediction_file(outfolder, pred_new, dimensions, prefix, n)
    ## Support vector regression
    outfolder = 'predictions_svr'
    for prefix, num in zip(prefixes,num_sub):
        for n in range(num):
            pred_s = np.empty((num_steps,len(dimensions),len(predictions)))
            for ind in range(len(predictions)):
                pred_n = read_predictions(predictions[ind], prefix, n, dimensions)
                pred_s[:,:,ind] = pred_n
            pred_new = np.empty((num_steps,len(dimensions)))
            for dim in range(len(dimensions)):
                pred_s_context = contextualise( pred_s, context_opt[dim], context_opt[dim] )
                pred_new[:,dim] = svr_models[dim].predict(pred_s_context.reshape(pred_s_context.shape[0],-1))
            write_prediction_file(outfolder, pred_new, dimensions, prefix, n)
#


def contextualise(pred_models, context_past=5, context_future=5):
    context_past   = int(context_past)
    context_future = int(context_future)
    factor = 1 + context_past + context_future
    pred_models_context = np.empty((pred_models.shape[0],pred_models.shape[1],pred_models.shape[2]*factor))
    for m in range(pred_models.shape[0]):
        n = 0
        pred_models_context[m,:,pred_models.shape[2]*n:pred_models.shape[2]*(n+1)] = pred_models[m,:,:]  # self
        for c in range(context_past):  # past
            n = 1 + c
            ind = m-c
            ind = np.max([0,ind])
            pred_models_context[m,:,pred_models.shape[2]*n:pred_models.shape[2]*(n+1)] = pred_models[ind,:,:]
        for c in range(context_future):  # future
            n = 1 + context_past + c
            ind = m+c
            ind = np.min([pred_models.shape[0]-1,ind])
            pred_models_context[m,:,pred_models.shape[2]*n:pred_models.shape[2]*(n+1)] = pred_models[ind,:,:]
    return pred_models_context


def read_gold_predictions(submission_path, prefixes, num_seqs, gold_standard_path, dimensions):
    if submission_path[-1]!='/':    submission_path += '/'
    if gold_standard_path[-1]!='/': gold_standard_path += '/'
    gold = np.empty((0,len(dimensions)))
    pred = np.empty((0,len(dimensions)))
    for prefix, num_seq in zip(prefixes, num_seqs):
        for n in range(0, num_seq):
            filename = prefix + '_' + str(n+1).zfill(2) + '.csv'
            # Load gold standard
            gold_n  = pd.read_csv(gold_standard_path + filename, sep=';', usecols=dimensions).values
            gold    = np.concatenate((gold, gold_n), axis=0)
            pred_n  = pd.read_csv(submission_path + filename, sep=';', usecols=dimensions).values
            if pred_n.shape[0] > gold_n.shape[0]:
                pred_n = pred_n[:gold_n.shape[0],:]
            elif pred_n.shape[0] < gold_n.shape[0]:
                missing = gold_n.shape[0]-pred_n.shape[0]
                print('Warning: Missing predictions in file ' + submission_path + filename + ' - repeating last prediction ' + str(missing) + ' times!')
                pred_n = np.concatenate((pred_n, np.repeat(pred_n[-1],missing)), axis=0)
            pred = np.concatenate((pred, pred_n), axis=0)
    return pred, gold


def read_predictions(submission_path, prefix, n, dimensions):
    if submission_path[-1]!='/':    submission_path += '/'
    filename = prefix + '_' + str(n+1).zfill(2) + '.csv'
    pred_n = pd.read_csv(submission_path + filename, sep=';', usecols=dimensions).values
    return pred_n


def write_prediction_file(path_output, predictions, targets, prefix='Test_DE_', n=1, labels_per_sec=10):
    if path_output[-1]!='/': path_output += '/'
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    seq_len = predictions.shape[0]
    # add time stamp
    time_stamp = np.linspace(0., (seq_len-1)/float(labels_per_sec), seq_len).reshape(-1,1)
    pred_inst  = np.concatenate( (time_stamp, predictions), axis=1 )
    # create data frame and write file
    instname = prefix + '_' + str(n+1).zfill(2)
    filename = path_output + instname + '.csv'
    data_frame         = pd.DataFrame(pred_inst, columns=['timestamp']+targets)
    data_frame['name'] = '\'' + instname + '\''
    data_frame.to_csv(filename, sep=';', columns=['name','timestamp']+targets, index=False, float_format='%.6f')


if __name__ == '__main__':
    main()
