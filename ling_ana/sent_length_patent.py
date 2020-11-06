import pandas as pd
import numpy as np
import sys
from definitions import ROOT_DIR


def get_sentences(text: str):
    """
    Splits a text into separate sentences
    :param text: string
    :return: list of sentences
    """
    # Splits at paragraphs
    paragraphs = [p for p in text.split('\n') if p]
    # Splits at long whitespaces (9)
    paragraphs = [word for p in paragraphs for word in p.split('         ')]
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(nltk.sent_tokenize(paragraph))
    return sentences


def count_sentence_length(sentences: list):
    """
    counts the numbers of word per sentence, ACHTUNG: Bei claim werden 10 . auch als satz der länge 2 gezählt :(
    :param text: string
    :return: list containing the number of words per sentence
    """
    return [len(sentence.split()) for sentence in sentences]


def get_sentence_lengths_column(column: pd.Series):
    """
    counts the sentence length for all sentences in a column of a dataframe
    :param column: pd.Series
    :return: list of sentence lengths for all sentences in column
    """
    leng = []
    sent = []
    for i in range(len(column)):
        sentences = get_sentences(column[i])
        lengths = count_sentence_length(sentences)
        sent.extend(sentences)
        leng.extend(lengths)
    return sent, leng


def remove_short_sentence_blocks(sentences: list, lengths: list):
    """
    removes sentence blocks with the method that the current sentence is removed if the current and either the next 2
    or the preceding 2 sentences are all shorter than 10 words.
    Idea: Should i just remove all sentences with length < 10? Dann ist auch das Problem mit den claims weg
    :param sentences: list of sentences
    :param lengths: list of lengths of sentences
    :return: shortened list of sentences and their corresponding lengths
    """
    df = pd.DataFrame({'sentences': sentences, 'lengths': lengths})
    index_to_drop = list(df[df['lengths'] < 10].index)
    block_index = []
    for i in index_to_drop:
        # Checks if previous next 2 or the preceding 2 sentences are also shorter than 10 words and therefore in the index_to_drop
        if (i + 1 in index_to_drop and i + 2 in index_to_drop) or (
                i - 1 in index_to_drop and i - 2 in index_to_drop):  # or (i-1 in index_to_drop and i+1 in index_to_drop)
            block_index.append(i)
    df.drop(block_index, inplace=True)
    return list(df['sentences']), list(df['lengths'])


def remove_short_sentences(sentences: list, lengths: list, limit: int):
    """
    removes short sentences (shorter than 10 words) from list
    :param sentences: list of sentences
    :param lengths: list of lengths
    :return: shortened list of sentences and list of lengths
    """
    df = pd.DataFrame({'sentences': sentences, 'lengths': lengths})
    # If the length is smaller than the limit, this row wants to be removed from the dataframe
    index_to_drop = df[df['lengths'] < limit].index
    df.drop(index_to_drop, inplace=True)
    return list(df['sentences']), list(df['lengths'])


def remove_long_sentences(sentences: list, lengths: list, limit: int):
    """
    removes long sentences (longer than 250 words) from list
    :param sentences: list of sentences
    :param lengths: list of lengths
    :return: shortened list of sentences and list of lengths
    """
    df = pd.DataFrame({'sentences': sentences, 'lengths': lengths})
    # If the length is bigger than the limit, this row wants to be removed from the dataframe
    index_to_drop = df[df['lengths'] > limit].index
    df.drop(index_to_drop, inplace=True)
    return list(df['sentences']), list(df['lengths'])


def print_long_sentences(sentences: list, lengths: list, limit: int):
    """
    Prints the index, length and its text of extreme long sentences (longer than 200 words)
    :param sentences: list of sentences
    :param lengths: list of lengths
    :return: None
    """
    long_sentences_index = [idx for idx, val in enumerate(lengths) if val >= limit]
    for i in long_sentences_index:
        print('Sentence with index {0} and length {1}'.format(i, lengths[i]))
        print(sentences[i])
        print('\n\n')


def plot_hist_2_distributions(dis1: np.array, name_dis1: str, dis2: np.array, name_dis2: str, xlabel: str):
    """
    plots 2 distributions of frequencies
    :param dis1: np.array containing the numbers for distribution 1
    :param name_dis1: Name of the distribution 1
    :param dis2: np.array containing the numbers for distribution 2
    :param name_dis2: Name of the distribution 2
    :param xlabel: x-axis label
    :return: plots and saves a figure containing both distributions
    """
    plt.figure()
    sns.distplot(dis1, color="green", bins="doane", hist_kws={"align": "left"}, label=name_dis1) #skyblue?
    sns.distplot(dis2, color="blue", bins="doane", hist_kws={"align": "left"}, label=name_dis2)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    file_name = os.path.join(ROOT_DIR,'plots/histogram_2_distributions_{2}_{0}_vs_{1}.svg'.format(name_dis1, name_dis2, xlabel))
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    plt.savefig(file_name)
    plt.show()


def plot_hist_sentence_length_comparison(column_name: str, lengths_wo_long, lengths_short_wo_long, lengths_short_wo_long_wo12):
    """
    Sentence removal analysis depending on sentence length
    :param column_name:
    :param lengths_wo_long:
    :param lengths_short_wo_long:
    :param lengths_short_wo_long_wo12:
    :return:
    """
    plt.figure()
    sns.distplot(np.array(lengths_wo_long), bins="doane", hist_kws={"align": "left"}, color="skyblue",
                 label='with short sentence blocks')
    sns.distplot(np.array(lengths_short_wo_long), bins="doane", hist_kws={"align": "left"}, color="red",
                 label='wo short sentence blocks')
    sns.distplot(np.array(lengths_short_wo_long_wo12), bins="doane", hist_kws={"align": "left"}, color="olive",
                 label='wo short sentence blocks, wo sentences 1/2 words long')
    plt.xlabel('Sentence length claim')
    plt.ylabel("Frequency")
    plt.legend()
    file_name = os.path.join(ROOT_DIR, 'plots/histogram_3_distributions_{0}_sentence_length.svg',format(column_name))
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    plt.savefig(file_name)
    plt.show()


def sentence_length_analysis(df: pd.DataFrame, column_name:str):
    """
    Does a sentence length analysis with different plots, extreme long sentences are longer than 512, sentences shorter
    than 3 are removed, and short sentence blocks are removed and then analyzed
    :param df: dataframe containing text data
    :param column_name: name of the column of the dataframe which needs to be analyzed
    :return: None
    """
    # Plot sentence length for columns_name
    sentences, lengths = get_sentence_lengths_column(df[column_name])
    plot_hist(np.array(lengths), "Sentence lengths {0}".format(column_name))
    sentences_wo_long, lengths_wo_long = remove_long_sentences(sentences, lengths)

    # Sentences with extreme length
    sent_extreme, len_extreme = remove_short_sentences(sentences, lengths, 512)
    plot_hist(np.array(len_extreme), "Sentence lengths {0} of extreme long sentences".format(column_name))

    sentences_short_wo_long, lengths_short_wo_long = remove_short_sentence_blocks(sentences_wo_long, lengths_wo_long)

    plot_hist_2_distributions(np.array(lengths_wo_long), "with short sentence blocks",
                              np.array(lengths_short_wo_long), "without short sentence blocks",
                              "Sentence length {0}".format(column_name))

    sentences_short_wo_long_wo12, lengths_short_wo_long_wo12 =\
        remove_short_sentences(sentences_short_wo_long, lengths_short_wo_long, 3)

    plot_hist(np.array(lengths_short_wo_long), "Sentence Length {0} without short sentence blocks".format(column_name))

    plot_hist_sentence_length_comparison(column_name, lengths_wo_long, lengths_short_wo_long, lengths_short_wo_long_wo12)


def main(ROOT_DIR, file_name):
    """
    Analyze the sentence length for columns abstract, claim and description
    :param ROOT_DIR: root directory of data folder
    :param file_name: file name of the csv file
    :return:
    """
    df = pd.read_csv(os.path.join(ROOT_DIR, file_name))
    df = df.head(1000)
    sentence_length_analysis(df, 'abstract')
    sentence_length_analysis(df, 'claim')
    sentence_length_analysis(df, 'description')


if __name__ == '__main__':
    file_name = sys.argv[1]
    # file_name = 'data/patent/part-000000000674.csv'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name))


