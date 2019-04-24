#!/usr/bin/python2
# python2.7
# Author: Maximilian Schmitt, 2017-2018

import numpy as np


def load_labels(filename, col_labels=1, skip_header=True, delim=';'):
    # Reads column col_labels from csv file (arbitrary data type)
    # filename:      csv-filename
    # col_labels:    Index of the target column (indexing starting with 1, default: 1)
    # skip_header:   Skip the first line of the given csv file (default: True)
    # delim:         Delimiter (default: ';')
    
    labels = []
    with open(filename, 'r') as csv_file:
        if skip_header:
            next(csv_file)
        for line in csv_file:
            cols = line.split(delim)
            labels.append(cols[col_labels-1].rstrip())
    return np.array(labels)


def load_features(filename, skip_header=True, skip_instname=True, delim=';', num_lines=0):
    # Reads a csv file (only numbers, except for first item if skip_instname=True) with given delimiter
    # filename:      csv-filename
    # skip_header:   Skip the first line of the given csv file (default: True)
    # skip_instname: Skip the first column/attribute of the given csv file, e.g., the filename (default: True)
    # delim:         Delimiter (default: ';')
    # num_lines:     Number of lines in the CSV file. If given, the function is faster.
    # 
    # Return: numpy array (float)
    # 
    # This function is 6.3 times faster than loadtxt (also without number of lines given):
    #  data = np.loadtxt(open(filename, 'r'), delimiter=delim)
    
    if num_lines==0:
        num_lines = get_num_lines(filename,skip_header)
    
    data = np.empty((num_lines,get_num_columns(filename,skip_header,skip_instname,delim)), float)
    
    with open(filename, 'r') as csv_file:
        if skip_header:
            next(csv_file)
        c = 0
        for line in csv_file:
            offset = 0
            if skip_instname:
                offset = line.find(delim)+1
            data[c,:] = np.fromstring(line[offset:], dtype=float, sep=delim)
            c += 1
    
    return data


# Helper functions
def get_num_lines(filename,skip_header):
    with open(filename, 'r') as csv_file:
        if skip_header:
            next(csv_file)
        c = 0
        for line in csv_file:
            c += 1
    return c

def get_num_columns(filename,skip_header,skip_instname,delim):
    with open(filename, 'r') as csv_file:
        if skip_header:
            csv_file.readline()
        line = csv_file.readline()
        offset = 0
        if skip_instname:
            offset = line.find(delim)+1
        cols = np.fromstring(line[offset:], dtype=float, sep=delim)
    return len(cols)

