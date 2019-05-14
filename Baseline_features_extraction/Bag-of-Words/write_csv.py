#!/usr/bin/python
# Author: Maximilian Schmitt, 2017-2019

import os.path
import numpy as np


def save_features(filename,
                  data,
                  append=False,
                  instname='',
                  header='',
                  delim=';',
                  precision=8,
                  first_time=False):
    # Write a csv file
    # filename:   csv-filename
    # data:       Data given as numpy array
    # append:     If True, file (filename) is appended, otherwise, it is overwritten
    # instname:   If non-empty string is given, instname is added as a first element in each row
    # header:     If non-empty string is given, print header line as a string
    # delim:      Delimiter (default: ';')
    # precision:  Floating point precision (default: 8)
    # first_time: If True, first column has precision of only 2 decimals

    mode = 'w'
    if append:
        mode = 'a'
        if os.path.isfile(filename):
            header = ''  # do never write header if file already exists and is appended

    with open(filename, mode) as csv_file:
        if len(header) > 0:
            csv_file.write(header + '\n')

        for row in data:
            if len(instname) > 0:
                csv_file.write('\'' + instname + '\'' + delim)
            if first_time:
                csv_file.write(
                    np.array2string(np.array([row[0]]),
                                    max_line_width=10,
                                    precision=2,
                                    floatmode='fixed',
                                    separator=delim)[1:-1].replace(' ', '') +
                    delim)
                csv_file.write(
                    np.array2string(row[1:],
                                    max_line_width=100000,
                                    precision=precision,
                                    separator=delim)[1:-1].replace(' ', '') +
                    '\n')  # check max_line_width
            else:
                csv_file.write(
                    np.array2string(row,
                                    max_line_width=100000,
                                    precision=precision,
                                    separator=delim)[1:-1].replace(' ', '') +
                    '\n')  # check max_line_width
