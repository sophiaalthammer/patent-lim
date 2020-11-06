import pandas as pd
import spacy
import os
import sys
from ling_ana.count_words_patent import get_word_count_column
from definitions import ROOT_DIR


def get_hyphen_exp(text: str, nlp):
    """
    Returns a list of all hyphen expressions contained in the text
    :param text: string
    :param nlp: spacy language model
    :return: list of hyphen expressions contained in text
    """
    doc = nlp(text)
    token_list = [token.text for token in doc]

    hyphen_expressions = []
    for i in range(len(token_list)):
        if token_list[i] == '-':
            hyphen_expressions.append(token_list[i - 1] + token_list[i] + token_list[i + 1])
    return hyphen_expressions


def get_hyphen_exp_column(column: pd.Series, nlp):
    """
    Gets all the hyphen expressions contained in a pd.Series of text
    :param column: pd.Series containing text in each element
    :param nlp: spacy language model
    :return: list of all hyphen expressions contained in the text of the column cells
    """
    hyphen_expressions = []
    for i in range(len(column)):
        conj = get_hyphen_exp(column[i], nlp)
        hyphen_expressions.extend(conj)
    return hyphen_expressions


def get_share_hyphen_exp(column: pd.Series, nlp):
    word_counts = get_word_count_column(column)
    hyphen_expressions = get_hyphen_exp_column(column, nlp)
    return (len(hyphen_expressions) * 2) / sum(word_counts)


def main(ROOT_DIR, file_name):
    """
    Prints the share of the hypen expressions in the claims and the descriptions
    :param ROOT_DIR: root directory of data folder
    :param file_name: file name of the pickle file
    :return:
    """
    df = pd.read_pickle(os.path.join(ROOT_DIR, file_name))
    df = df.head(100)

    nlp = spacy.load("en_core_web_sm")
    print('Share of hyphen expressions in patent claims: {0}'.format(get_share_hyphen_exp(df['claim'], nlp)))
    print('Share of hyphen expressions in patent claims: {0}'.format(get_share_hyphen_exp(df['description'], nlp)))


if __name__ == '__main__':
    file_name = sys.argv[1]
    # file_name = 'data/patent/part-000000000674_preprocessed.pkl'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name))
