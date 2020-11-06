import pandas as pd
from ling_ana import patent_vocab_train
from ling_ana import sent_length_patent
from ling_ana import patent_vocab_ana
from transformers import BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from definitions import ROOT_DIR


def length_sp_model(column: pd.Series, model_prefix):
    """
    Calculates for every text given in the cells of the column the length of the text encoded in a sentencepiece model
    which is specified with model_prefix
    :param column: pd.Series
    :param model_prefix: name of the model
    :return: list of lengths in sentencepiece encoding
    """
    lengths = []
    for text in column:
        lengths.append(len(patent_vocab_train.sp_encode(text, model_prefix)))
    return lengths


def length_bert(column: pd.Series, tokenizer: BertTokenizer):
    """
    Calculates for every text given in the cells of the column the length of the text encoded in a BERT base cased model
    :param column: pd.Series
    :param bert_tokenizer: BertTokenizer
    :return: list of lengths in BERT encoding
    """
    lengths = []
    for text in column:
        lengths.append(len(tokenizer.tokenize(text)))
    return lengths


def lengths_4_encodings(column: pd.Series, model_prefix: str, bert_tokenizer: BertTokenizer, scibert_tokenizer: BertTokenizer):
    """
    Calulates for all text in a column the lengths of the text in word count, sentencepiece encoding, bert encoding and
    scibert encoding
    :param column: pd.Series
    :param model_prefix: name of the sentencepiece model
    :param bert_tokenizer: BertTokenizer
    :param scibert_tokenizer: BertTokenizer containing the SciBert
    :return: a dataframe containing the lengths of the texts in word count, sentencepiece encoding
    """
    sentences, lengths_words = sent_length_patent.get_sentence_lengths_column(column)
    lengths_patent_vocab = length_sp_model(sentences, model_prefix)
    lengths_bert = length_bert(sentences, bert_tokenizer)
    lengths_scibert = length_bert(sentences, scibert_tokenizer)
    df = pd.DataFrame(zip(lengths_words, lengths_patent_vocab, lengths_bert, lengths_scibert))
    df.rename(columns={0: 'Word count', 1: 'Patent Vocab', 2: 'Bert Vocab', 3: 'SciBert Vocab'}, inplace=True)
    return df


def plot_kde(title: str, xlabel: str, df: pd.DataFrame, colours: list, zoom_in=False):
    """
    Plots for every column in the dataframe a kde plot with the name of the column and the colour given in colours
    :param title: Title of the plot
    :param xlabel: name of the x-axis
    :param df: Dataframe containing the numbers to be plotted, plots for each column one line
    :param colours: colours, for each column there has to be one colour
    :param zoom_in: by default False, if True then it displays the x-axis range from 0 to 200
    :return: plot and saves plot
    """
    plt.figure()
    for i in range(len(df.columns)):
        plot = sns.kdeplot(df.iloc[:, i], color=colours[i], label=df.columns[i])
        plt.axvline(x=df.iloc[:, i].mean(), color=colours[i], linestyle='--')
        print(df.iloc[:, i].mean())
        if zoom_in:
            plot.set_xlim([0,200])     # Zoom in into plot
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    #plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR,'plots/kdeplot_{1}_{0}.svg'.format(xlabel, title)), bbox_inches='tight',pad_inches = 0)
    plt.show()


def split_ratio_encoding_model(column: pd.Series, models: list):
    """
    Computes the split ratio for all text contained in column for all models contained in the model list
    sentencepiece models must be named by a string, bert tokenizer must be given in the list of models
    :param column: pd.Series containing the text
    :param models: list of models used to encode, sentencepiece models must be named by a string, bert tokenizer must
    be given in list
    :return: list of split ratios for different encodings
    """
    split_ratio = []
    for i in range(len(models)):
        split_ratio.append(patent_vocab_ana.split_ratio_encoding(column, models[i]))
    return split_ratio


def split_ratio_encoding_multiple_columns(df: pd.DataFrame, column_names: list, models: list, model_names):
    """
    Returns a dataframe containing the split ratios for all column determined in column_names of the dataframe and
    for all models contained in list of models (sentencepiece models must be named by a string, bert tokenizer must
    be given in the list of models)
    :param df: pd.Dataframe containing the texts
    :param column_names: list of column names (str) which should be analyzed and which contain text
    :param models: list of models, sentencepiece models must be named by a string, bert tokenizer must be given in
    the list of models
    :param model_names: names of models contained in the models list, needs to have the same length as models
    :return: dataframe containing the split ratios for all models and for all column given in column_names
    """
    split_ratios = []
    for column_name in column_names:
        sr = split_ratio_encoding_model(df[column_name], models)
        split_ratios.append(sr)
    return pd.DataFrame(split_ratios, columns=model_names, index=column_names)


def main(ROOT_DIR: str, file_name: str, model_dir: str, bert_tokenizer: str, scibert_tokenizer: str):
    df = pd.read_pickle(os.path.join(ROOT_DIR, file_name))
    df = df.head(100)

    # column of dataframe which is to be analyzed
    column = df['text']
    # model prefix of sentencepiece model
    model_prefix = os.path.join(ROOT_DIR,model_dir)
    bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(ROOT_DIR,bert_tokenizer))
    scibert_tokenizer = BertTokenizer.from_pretrained(os.path.join(ROOT_DIR, scibert_tokenizer))

    # Plot sentence length encodings
    lengths = lengths_4_encodings(column, model_prefix, bert_tokenizer, scibert_tokenizer)
    colours4 = ['orange', 'blue', 'green', 'red']
    plot_kde('Comparison for vocabulary encodings on full text', 'Sentence length', lengths, colours4, zoom_in=True)



if __name__ == '__main__':
    file_name = sys.argv[1]
    model_dir = sys.argv[2]
    bert_tokenizer = sys.argv[3]
    scibert_tokenizer = sys.argv[4]
    # file_name = 'data/patent/part-000000000674_preprocessed.pkl'
    # model_dir = 'models/sentencepiece/patents_part-000000000674_sp_preprocessed/patent_wdescr_5m_sent_30k_vocab'
    # bert_tokenizer = ,'bert-base-cased'
    # scibert_tokeniezr = 'models/scibert_scivocab_uncased'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name), str(model_dir))





