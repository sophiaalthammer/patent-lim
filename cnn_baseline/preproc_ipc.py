# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import sys
import os
import errno
import pandas as pd
from definitions import ROOT_DIR


def main(file_number: str, input_dir: str, output_dir: str):
    df = pd.read_csv(os.path.join(ROOT_DIR, '{0}train_0{1}.tsv'.format(input_dir, file_number)),
                     sep='\t',
                     header=None)

    # column 7 is ipc tag
    # column 3 is claim, split claim and take first 128 words
    df[3] = df[3].str.split()

    for i in range(len(df)):
        filename = os.path.join(ROOT_DIR, '{0}{1}/doc_{2}_{3}.txt'.format(output_dir, df[7].iloc[i], file_number, i))
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(os.path.join(ROOT_DIR, '{0}{1}/doc_{2}_{3}.txt'.format(output_dir, df[7].iloc[i], file_number, i)),
                  'w') as f:
            f.write(' '.join(df[3].iloc[i][0:128]))


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    start_files = sys.argv[3]
    end_files = sys.argv[4]
    # input_dir = /data/ipc_classification/'
    # output_dir = 'emnlp_baseline3/data/ipc/20k/'
    files = ["{0:03}".format(i) for i in range(start_files, end_files)]
    for file in files:
        print(file)
        main(file, str(output_dir))