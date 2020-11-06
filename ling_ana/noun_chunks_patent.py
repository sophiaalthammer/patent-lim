import pandas as pd
import numpy as np
import string
import spacy
import os
import sys
from ling_ana.ipc_cpc_class_occurences import plot_hist
from ling_ana import count_words_patent
from ling_ana.sent_length_patent import remove_long_sentences
from definitions import ROOT_DIR


def get_noun_chunks(text: string, nlp):
    """
    Gets the noun chunks in a text (flat phrases that have a noun as their head)
    :param text: string
    :param nlp: spacy language model
    :return: list of noun chunks found in the text
    """
    doc = nlp(text)
    return list(doc.noun_chunks)


def get_span_length(spans: list):
    """
    For spacy.span object in a list gets the length of the object
    :param spans: list of spacy.span objects
    :return: list of lengths of the spacy.span objects
    """
    return [len(span) for span in spans]


def get_share_long_spans(span_length: list):
    """
    gets the share of long spans (spans longer than 4) to all spans
    :param span_length: list of lengths of the spans
    :return: percentage of long spans in all spans
    """
    return sum(length >= 4 for length in span_length)/len(span_length)


def get_noun_chunk_column(column: pd.Series, nlp):
    """
    Gets the noun chunks for the text contained in a pd.Series, if the noun chunks are longer than 10 we assume it
    is not a noun phrase any more but a enumeration and we drop it
    :param column: pd.Series containing the text to be analyzed
    :param nlp: spacy language model
    :return: list of chunks and list of their lengths
    """
    chunks = []
    lengths = []
    for i in range(len(column)):
        chunk = get_noun_chunks(column[i], nlp)
        chunks.extend(chunk)
        lengths.extend(get_span_length(chunk))
    chunks, lengths = remove_long_sentences(chunks, lengths, 10)
    return chunks, lengths


def get_share_noun_chunks_column(column: pd.Series, nlp):
    """
    Calculates the share of words in noun phrases to all words in text
    :param column: pd.Series containing the text to be analyzed
    :param nlp: spacy language model
    :return: precentage of noun phrase words to all words in text
    """
    counts = count_words_patent.get_word_count_column(column)
    chunks, lengths = get_noun_chunk_column(column, nlp)
    #chunks, lengths = remove_long_sentences(chunks, lengths, 10)
    return sum(lengths)/sum(counts)


def hist_noun_chunks_column(df: pd.DataFrame, number_docs: int, column_name: str, nlp):
    """
    Prints the share of the noun chunks in the column and plots a histpogram of the nou chunk length in the  column
    :param df: dataframe with text data
    :param number_docs: number of rows/documents of the dataframe to be considered
    :param column_name: name of the column
    :param nlp: Spacy language model
    :return: None
    """
    df = df.head(number_docs)

    share = get_share_noun_chunks_column(df[column_name], nlp)
    print('Share of noun chunks in column {0}: {1}'.format(column_name,share))

    # get noun chunks and lengths
    noun_chunks, noun_chunks_lengths = get_noun_chunk_column(df[column_name], nlp)

    # remove noun chunks longer than 10 because they are enumerations
    noun_chunks2, noun_chunks_lengths2 = remove_long_sentences(noun_chunks, noun_chunks_lengths, 10)

    # plot noun chunk length distribution for column
    plot_hist(np.array(noun_chunks_lengths2), "Noun Chunk Length {0}".format(column_name), 'Noun Chunk Length')


def main(ROOT_DIR, file_name):
    """
    Analyzed noun chunk distribution for title, abstract, claim, description and full test of patents,
    requires a pickle file as input with a dataframe containing preprocessed patent text
    :param ROOT_DIR: root directory of data folder
    :param file_name: file name of the pickle file
    :return:
    """
    df = pd.read_pickle(os.path.join(ROOT_DIR, file_name))
    # Load spacy language model
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1500000

    # Noun chunk length in Title
    hist_noun_chunks_column(df, 1000, 'title', nlp)
    hist_noun_chunks_column(df, 1000, 'abstract', nlp)
    hist_noun_chunks_column(df, 100, 'claim', nlp)
    hist_noun_chunks_column(df, 100, 'description', nlp)
    hist_noun_chunks_column(df, 100, 'text', nlp)


if __name__ == '__main__':
    file_name = sys.argv[1]
    #file_name = 'data/patent/part-000000000674.csv'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name))






