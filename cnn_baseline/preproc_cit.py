import os
import sys
import pandas as pd
from definitions import ROOT_DIR


def main(ROOT_DIR: str, file_name: str, number_of_samples: int, output_dir: str):
    """
    Creates the negative and positive text files for citation prediction baseline
    :param ROOT_DIR:
    :param file_name:
    :param number_of_samples:
    :param output_dir:
    :return:
    """
    df = pd.read_pickle(os.path.join(ROOT_DIR, file_name))

    df = df.head(number_of_samples)

    # separate positive and negatvies
    df_pos = df[df['label'] == 1].reset_index(drop=True)
    df_neg = df[df['label'] == 0].reset_index(drop=True)

    # write in one line each pair (128 tokens of first and 128 tokens of last)
    df_pos['pub_claim'] = df_pos['pub_claim'].str.split()
    df_pos['cit_claim'] = df_pos['cit_claim'].str.split()
    df_neg['pub_claim'] = df_neg['pub_claim'].str.split()
    df_neg['cit_claim'] = df_neg['cit_claim'].str.split()

    # Save positive and negative citation pairs in output_dir
    with open(os.path.join(ROOT_DIR, '{0}/cit_pos.txt'.format(output_dir)), 'w') as f:
        for i in range(len(df_pos)):
            f.write(' '.join(df_pos['pub_claim'].iloc[i][0:128]) + ' [SEP] ' + ' '.join(
                df_pos['cit_claim'].iloc[i][0:128]) + '\n')

    with open(os.path.join(ROOT_DIR, '{0}/cit_neg.txt'.format(output_dir)), 'w') as f:
        for i in range(len(df_neg)):
            f.write(' '.join(df_neg['pub_claim'].iloc[i][0:128]) + ' [SEP] ' + ' '.join(
                df_neg['cit_claim'].iloc[i][0:128]) + '\n')


if __name__ == '__main__':
    file_name = sys.argv[1]
    number_of_samples = sys.argv[2]
    output_dir = sys.argv[3]
    # file_name = 'data/citations/patent-contents-for-citations-wclaims/citations-only-type-x-with_claims_eval_data.pkl'
    # number_of_samples = 16000
    # output_dir = 'cnn_baseline/data/citation/'
    main(ROOT_DIR, str(file_name), int(number_of_samples), str(output_dir))