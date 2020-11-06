import pandas as pd
import numpy as np
import spacy
import os
import sys
from ling_ana.noun_chunks_patent import get_noun_chunk_column
from ling_ana.count_words_patent import get_word_count_column
from ling_ana.ipc_cpc_class_occurences import plot_hist
from ling_ana.sent_length_patent import get_sentence_lengths_column, remove_long_sentences
from ling_ana.hyphen_exp import get_share_hyphen_exp, get_hyphen_exp_column
from definitions import ROOT_DIR


def main(ROOT_DIR, file_name):
    """
    Does the linguistic analysis for preprocessed semantic Scholar text for a pickle file with a dataframe with
    a column 'title' and a column 'abstract'
    :param ROOT_DIR:
    :param file_name:
    :return:
    """
    df = pd.read_pickle(os.path.join(ROOT_DIR, file_name))

    # Word counts for title and abstract
    word_counts_title = get_word_count_column(df['title'])
    print(np.mean(word_counts_title))
    plot_hist(np.array(word_counts_title), "Word Counts", 'Semantic Scholar Title')

    word_counts_abstract = get_word_count_column(df['abstract'])
    print(np.mean(word_counts_abstract))
    plot_hist(np.array(word_counts_abstract), "Word Counts", 'Semantic Scholar Abstract')

    # Sentence lengths for abstract, long ones removed as longer than 250 is not a sentence any more
    sentences, lengths = get_sentence_lengths_column(df['abstract'])
    sent_short, leng_short = remove_long_sentences(sentences, lengths, 250)
    plot_hist(np.array(leng_short), "Sentence Lengths", 'Semantic Scholar Abstract')

    # Noun chunks for titles and abstracts
    nlp = spacy.load("en_core_web_sm")
    head = df.head(1000)
    noun_chunks_ab, noun_chunks_lengths_ab = get_noun_chunk_column(head['abstract'], nlp)
    plot_hist(np.array(noun_chunks_lengths_ab), "Semantic Scholar Noun Chunks Lengths Abstract", 'Noun chunk lengths')

    noun_chunks_tit, noun_chunks_lengths_tit = get_noun_chunk_column(head['title'], nlp)
    plot_hist(np.array(noun_chunks_lengths_tit), "Semantic Scholar Noun Chunks Lengths Title", 'noun chunk lengths')

    # Hyphen words
    # Calculate share with multiplying the hyphen expressions by 3 because word counts
    # counts them as 3 words
    print('Share of hyphen expressions in Semantic Scholar abstracts: {0}'.format(
        get_share_hyphen_exp(head['abstract'], nlp)))
    hyphen_exp = get_hyphen_exp_column(head['abstract'], nlp)
    print('Hypen expressions: {0}'.format(hyphen_exp[0]))


if __name__ == '__main__':
    file_name = sys.argv[1]
    # file_name = 'data/semanticscholar/s2-corpus-000_clean.pkl'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name))