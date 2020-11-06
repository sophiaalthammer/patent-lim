import sentencepiece as spm
import pandas as pd
import regex as re
from ling_ana import count_words_patent
import os
import sys
from definitions import ROOT_DIR


def sp_preprocess(list_of_filenames: list):
    """
    Clean data so that just text of title, abstract and claims is left in csv for sentencepiece trainer
    :param list_of_filenames: list containing csv file names as strings
    :return: saves new csv files without descriptions under the same folder as csv files with description
    """
    for file in list_of_filenames:
        df = pd.read_csv(os.path.join(ROOT_DIR, file))
        df2 = df.drop(['publication_number', 'publication_date', 'filing_date', 'priority_date', 'title_truncated',
                       'abstract_truncated', 'claim_truncated', 'description_truncated', 'ipc', 'cpc'],
                      axis=1)  # 'description' if description should also be removed
        df2.to_csv(
            os.path.join(ROOT_DIR, '{0}_sp_preprocessed.csv'.format(file.split('.')[0]), index=False))


def sp_train(input: str, model_prefix: str, vocab_size: int, input_sentence_size: int):
    """
    Trains a sentencepiece model with the text file in input, vocabulary size in vocab_size and names it after
    model_prefix
    :param input: location and file name of the text file where the text to be trained on is contained
    :param model_prefix: name of the model
    :param vocab_size: integer value of size of the vocabulary
    :param input_sentence_size: number of sentences to take into account for training
    :return: trains and stores a sentencepiece model and its vocabulary in the folder of the project
    """
    spm.SentencePieceTrainer.Train('--input={0} --model_prefix={1} --vocab_size={2} '
                                   '--max_sentence_length=10000 --input_sentence_size={3} --shuffle_input_sentence=true '
                                   '--bos_piece=<CLS> --eos_piece=<SEP>'.format(input, model_prefix, vocab_size,
                                                                                input_sentence_size))


def sp_encode(text: str, model_prefix: str):
    """
    Encodes the given text with the model given with model_prefix
    :param text: str text which should be encoded
    :param model_prefix: name of the model
    :return: list of encoded text as sentencepiece
    """
    sp = spm.SentencePieceProcessor()
    sp.Load("{0}.model".format(model_prefix))
    return sp.EncodeAsPieces(text)


def number_text1_in_text2(text1: str, text2: str):
    """
    Computes the number of words that appear in text1 and text2 as well as the length of text1
    :param text1: str
    :param text2: str
    :return: number of words appearing in both texts, number of words total in text1
    """
    text1_splitted = re.split('-| ', count_words_patent.remove_punctuation(text1))
    text2_splitted = re.split('-| ', count_words_patent.remove_punctuation(text2))
    return len(set(text1_splitted) & set(text2_splitted)), len(text1_splitted)


def ratio_column1_in_column2(column1: pd.Series, column2: pd.Series):
    """
    Computes the ratio how much of the words in column1 are contained in column2 as well (if high (close to 1) then column1 is
     covered by column2, if low (close to 0) then column1 not covered by column2)
    :param column1: pd.Series containing text
    :param column2: pd.Series containing text
    :return: ratio of total overlaps in both columns to total number of words in both columns
    """
    if len(column1) == len(column2):
        number_words = []
        number_overlaps = []
        for i in range(len(column1)):
            number_overlaps.append(number_text1_in_text2(column1[i], column2[i])[0])
            number_words.append(number_text1_in_text2(column1[i], column2[i])[1])
    return sum(number_overlaps) / sum(number_words)


def unique_words(text: str):
    """
    returns a list of unique words contained in the given text
    :param text: str
    :return: list of unique words
    """
    return set(re.split('-| ', count_words_patent.remove_punctuation(text)))


def number_unique_words_df(df: pd.DataFrame):
    unique_words_row = set()
    for i in range(len(df['title'])):
        unique_words_row = unique_words_row.union(unique_words(df['title'].iloc[i]),
                                                  unique_words(df['abstract'].iloc[i]),
                                                  unique_words(df['claim'].iloc[i]),
                                                  unique_words(df['description'].iloc[i]))
    return len(unique_words_row)


def evaluate_overlap_words(df: pd.DataFrame):
    # Evaluate how many of the words in the certain columns are already contained in other ones (to see what set to
    # train on)
    print(ratio_column1_in_column2(df['title'], df['abstract']))  # 0.5562554872695347
    print(ratio_column1_in_column2(df['abstract'], df['claim']))  # 0.3391741909892959
    print(ratio_column1_in_column2(df['claim'], df['description']))  # 0.07265835232921353
    print(ratio_column1_in_column2(df['description'], df['claim']))  # 0.008607912767124979
    print(ratio_column1_in_column2(df['description'], df['abstract']))  # 0.00348835666086264
    print(ratio_column1_in_column2(df['abstract'], df['description']))  # 0.45333518164028935


def main(file_name: str, model_dir: str, vocab_size=30000, input_sentence_size=5000000):
    # Preprocess text for sentencepiece trainer
    sp_preprocess([file_name])
    # Take merged preprocessed csv file and train new sentencepiece model on it
    # NOTE: vocabulary: replace whitespace '_' in front with '' and where is '' replace with ## for training BERT!
    textfile_sp = os.path.join(ROOT_DIR, '{0}_sp_preprocessed.csv'.format(file_name.split('.')[0]))
    model_prefix = os.path.join(ROOT_DIR, model_dir)
    sp_train(textfile_sp, model_prefix, vocab_size, input_sentence_size)


if __name__ == '__main__':
    file_name = sys.argv[1]
    model_dir = sys.argv[2]
    # file_name = 'data/patent/part-000000000674.csv'
    # model_dir = 'models/sentencepiece/patents_part-000000000665_sp_preprocessed/patent_wdescr_50k_sent_4k_vocab'
    main(str(file_name), str(model_dir))

