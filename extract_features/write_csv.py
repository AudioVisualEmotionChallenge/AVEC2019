#!/usr/bin/python2
# python2.7
# Author: Maximilian Schmitt, 2017-2018

import os.path
import numpy as np

def save_features(filename, data, append=False, instname='', header='', delim=';', precision=8):
    # Write a csv file
    # filename: csv-filename
    # data:     Data given as numpy array
    # append:   If True, file (filename) is appended, otherwise, it is overwritten
    # instname: If non-empty string is given, instname is added as a first element in each row
    # header:   If non-empty string is given, print header line as a string
    # delim:    Delimiter (default: ';')
    # delim:    Floating point precision (default: 8)
    
    mode = 'w'
    if append:
        mode = 'a'
        if os.path.isfile(filename):
            header = ''  # do never write header if file already exists and is appended
    
    with open(filename, mode) as csv_file:
        if len(header)>0:
            csv_file.write(header + '\n')
        
        for row in data:
            if len(instname)>0:
                csv_file.write('\'' + instname + '\'' + delim)
            csv_file.write(np.array2string(row, max_line_width=100000, precision=precision, separator=delim)[1:-1].replace(' ','') + '\n')  # check max_line_width

