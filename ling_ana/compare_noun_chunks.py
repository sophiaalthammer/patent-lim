import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import os
import sys
import errno
from definitions import ROOT_DIR
from ling_ana import sent_length_patent
from ling_ana import noun_chunks_patent
from ling_ana.ling_ana_wiki import replace_hyphen_wiki


def plot_hist_2_distributions(dis1: np.array, dis1_name: str, dis2: np.array, dis2_name: str, xlabel: str, colour1: str, colour2: str):
    """
    plots 2 distributions of frequencies
    :param dis1: np.array containing the numbers for distribution 1
    :param dis1_name: Name of the distribution 1
    :param dis2: np.array containing the numbers for distribution 2
    :param dis2_name: Name of the distribution 2
    :param xlabel: x-axis label
    :return: plots and saves a figure containing both distributions
    """
    plt.figure()
    sns.distplot(dis1, color=colour1, label=dis1_name, kde_kws={'bw': 1},
                 hist_kws={'color':'white', 'edgecolor':colour1, 'linewidth':2, 'alpha':0.8}, kde=True)
    plt.axvline(x=np.mean(dis1), color=colour1, linestyle='--')
    sns.distplot(dis2, color=colour2, label=dis2_name, kde_kws={'bw': 1},
                 hist_kws={'color':'white', 'edgecolor':colour2, 'linewidth':2, 'alpha':0.8}, kde=True)
    plt.axvline(x=np.mean(dis2), color=colour2, linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    file_name = os.path.join(ROOT_DIR,'plots/histogram_2_distributions_{2}_{0}_vs_{1}.svg'.format(
        dis1_name, dis2_name,xlabel))
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    plt.savefig(file_name)
    plt.show()


def plot_hist_3_distributions(dis1: np.array, dis1_name: str, dis2: np.array, dis2_name: str, dis3: np.array, dis3_name: str,
                              xlabel: str, colour1: str, colour2: str, colour3: str):
    """
    plots 2 distributions of frequencies
    :param dis1: np.array containing the numbers for distribution 1
    :param dis1_name: Name of the distribution 1
    :param dis2: np.array containing the numbers for distribution 2
    :param dis2_name: Name of the distribution 2
    :param xlabel: x-axis label
    :return: plots and saves a figure containing both distributions
    """
    plt.figure()
    plt.hist([dis1, dis2, dis3], color=[colour1, colour2, colour3], label=['_nolegend_', '_nolegend_', '_nolegend_'],
             normed=True, alpha=0.6, align='left')
    plt.legend()
    bar1 = sns.distplot(dis1, color=colour1, label=dis1_name, kde_kws={'bw': 1}, hist=False,
                 hist_kws={'color':'1', 'edgecolor':colour1, 'linewidth':2, 'alpha':0.5, 'lw': 2}, kde=True)
    plt.axvline(x=np.mean(dis1), color=colour1, linestyle='--')
    bar2 = sns.distplot(dis2, color=colour2, label=dis2_name, kde_kws={'bw': 1}, hist=False,
                 hist_kws={'color':'1', 'edgecolor':colour2, 'linewidth':2, 'alpha':0.5, 'lw': 2}, kde=True)
    plt.axvline(x=np.mean(dis2), color=colour2, linestyle='--')
    bar3 = sns.distplot(dis3, color=colour3, label=dis3_name, kde_kws={'bw': 1},hist=False,
                 hist_kws={'color':'1', 'edgecolor': colour3, 'linewidth': 2, 'alpha': 0.5, 'lw': 2}, kde=True)
    plt.axvline(x=np.mean(dis3), color=colour3, linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    file_name = os.path.join(ROOT_DIR,'plots/histogram_3_distributions_{2}_{0}_vs_{1}_vs_{3}.svg'.format(
        dis1_name, dis2_name, xlabel, dis3_name))
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    plt.savefig(file_name, bbox_inches='tight',pad_inches = 0)
    plt.show()


def plot_hist_3(dis1: np.array, dis1_name: str, dis2: np.array, dis2_name: str, dis3: np.array, dis3_name: str,
                xlabel: str,  colour1: str, colour2: str, colour3: str):
    """
    Plots and saves a histogram of 3 distributions
    :param dis1: np.array with values of distribution1
    :param dis1_name: str with name of distribution1
    :param dis2: np.array with values of distribution2
    :param dis2_name: str with name of distribution2
    :param dis3: np.array with values of distribution3
    :param dis3_name: str with name of distribution3
    :param xlabel: name of distributions which are compared
    :return: plots and saves a figure containing both distributions
    """
    plt.figure()
    sns.distplot(dis1, color=colour1, label=dis1_name, kde_kws={'bw': 1}, kde=True)
    sns.distplot(dis2, color=colour2, label=dis2_name, kde_kws={'bw': 1}, kde=True)
    sns.distplot(dis3, color=colour3, label=dis3_name, kde_kws={'bw': 1}, kde=True)
    plt.axvline(x=np.mean(dis1), color=colour1, linestyle='--')
    plt.axvline(x=np.mean(dis2), color=colour2, linestyle='--')
    plt.axvline(x=np.mean(dis3), color=colour3, linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR,'plots/histogram_3_{0}.svg'.format(xlabel)))
    plt.show()


def main(ROOT_DIR: str, patent_loc: str, sem_loc: str, wiki_loc: str):
    """
    Compares the noun chunks of patent, semantic scholar and wikipedia text sections
    :param ROOT_DIR: root directory of data folder
    :param patent_loc: patent preprocessed pickle file
    :param sem_loc: semantic scholar preprocessed pickle file
    :param wiki_loc: wikipedia train raw data text file
    :return:
    """
    # Load patent data without short sentence blocks
    # Columns: Title, Abstract, Claim, Description
    patent_df = pd.read_pickle(
        os.path.join(ROOT_DIR, patent_loc))
    patent_df = patent_df.head(1000)

    # Load Semantic Scholar data cleaned (without empty cells, just english text in title and abstract)
    # Columns: Title, Abstract
    sem_df = pd.read_pickle(os.path.join(ROOT_DIR, sem_loc))
    sem_df = sem_df.head(1000)

    # Load Wiki data with captions/subcaptions
    # Text just entries
    location_output_file = os.path.join(ROOT_DIR, wiki_loc)
    wiki_text = open(location_output_file).read()

    # Compare noun phrases
    nlp = spacy.load("en_core_web_sm")

    # Patents
    patent_title_np, patent_title_np_leng = noun_chunks_patent.get_noun_chunk_column(patent_df['title'], nlp)
    patent_title_np_short, patent_title_np_leng_short = sent_length_patent.remove_long_sentences(
        patent_title_np, patent_title_np_leng, 10)

    # Semantic Scholar
    sem_title_np, sem_title_np_leng = noun_chunks_patent.get_noun_chunk_column(sem_df['title'], nlp)
    sem_title_np_short, sem_title_np_leng_short = sent_length_patent.remove_long_sentences(
        sem_title_np, sem_title_np_leng, 10)

    # Wiki
    wiki_sent = sent_length_patent.get_sentences(wiki_text)
    wiki_sent_hyphen_replaced = replace_hyphen_wiki(wiki_sent)
    wiki_np, wiki_np_leng = noun_chunks_patent.get_noun_chunk_column(wiki_sent_hyphen_replaced, nlp)
    wiki_np_short, wiki_np_leng_short = sent_length_patent.remove_long_sentences(wiki_np, wiki_np_leng, 10)

    # Plots
    plot_hist_3_distributions(np.array(patent_title_np_leng_short), 'Patent Titles',
                              np.array(wiki_np_leng_short), 'Wikipedia',
                              np.array(sem_title_np_leng_short), 'Semantic Scholar Titles',
                              'Noun chunk length',
                              'blue', 'green', 'red')
    plot_hist_2_distributions(np.array(patent_title_np_leng_short), 'Patent Titles',
                              np.array(sem_title_np_leng_short), 'Semantic Scholar Titles', 'Noun Phrase length'
                              , 'blue', 'red')

    # Patents Abstract
    patent_abs_np, patent_abs_np_leng = noun_chunks_patent.get_noun_chunk_column(patent_df['abstract'], nlp)
    patent_abs_np_short, patent_abs_np_leng_short = sent_length_patent.remove_long_sentences(patent_abs_np,
                                                                                             patent_abs_np_leng, 10)

    # Semantic Scholar Abstract
    sem_abs_np, sem_abs_np_leng = noun_chunks_patent.get_noun_chunk_column(sem_df['abstract'], nlp)
    sem_abs_np_short, sem_abs_np_leng_short = sent_length_patent.remove_long_sentences(sem_abs_np, sem_abs_np_leng, 10)

    # Plots
    plot_hist_2_distributions(np.array(patent_abs_np_leng_short), 'Patent Abstracts',
                              np.array(sem_abs_np_leng_short), 'Semantic Scholar Abstracts',
                              'Noun Phrase length', 'blue', 'red')

    plot_hist_2_distributions(np.array(patent_abs_np_leng_short), 'Patent Abstracts',
                              np.array(wiki_np_leng_short), 'Wikipedia',
                              'Noun Phrase length', 'blue', 'green')

    plot_hist_3_distributions(np.array(patent_abs_np_leng_short), 'Patent Abstracts',
                              np.array(wiki_np_leng_short), 'Wikipedia',
                              np.array(sem_abs_np_leng_short), 'Semantic Scholar Abstracts',
                              'Noun chunk length',
                              'blue', 'green', 'red')

    plot_hist_2_distributions(np.array(patent_title_np_leng_short), 'Patent Titles',
                              np.array(wiki_np_leng_short), 'Wikipedia', 'Noun Phrase length', 'blue', 'green')

    # Patents Claims
    patent_claim_np, patent_claim_np_leng = noun_chunks_patent.get_noun_chunk_column(patent_df['claim'], nlp)
    patent_claim_np_short, patent_claim_np_leng_short = sent_length_patent.remove_long_sentences(
        patent_claim_np, patent_claim_np_leng, 10)

    plot_hist_2_distributions(np.array(patent_claim_np_leng_short), 'Patent Claims',
                              np.array(wiki_np_leng_short), 'Wikipedia', 'Noun Phrase length', 'blue', 'green')

    plot_hist_3_distributions(np.array(patent_claim_np_leng_short), 'Patent Claims',
                              np.array(wiki_np_leng_short), 'Wikipedia',
                              np.array(sem_abs_np_leng_short), 'Semantic Abstracts',
                              'Noun Phrase length',
                              'blue', 'green', 'red')

    # Patent Descriptions
    patent_df2 = patent_df.head(100)
    patent_desc_np, patent_desc_np_leng = noun_chunks_patent.get_noun_chunk_column(patent_df2['description'], nlp)
    patent_desc_np_short, patent_desc_np_leng_short = sent_length_patent.remove_long_sentences(
        patent_desc_np, patent_desc_np_leng, 10)

    plot_hist_2_distributions(np.array(patent_desc_np_leng_short), 'Patent Descriptions',
                              np.array(wiki_np_leng_short), 'Wikipedia', 'Noun Phrase length', 'blue', 'green')

    # Full text
    patent_text_np, patent_text_np_leng = noun_chunks_patent.get_noun_chunk_column(patent_df2['text'], nlp)
    patent_text_np_short, patent_text_np_leng_short = sent_length_patent.remove_long_sentences(
        patent_text_np, patent_text_np_leng, 10)
    plot_hist_3_distributions(np.array(patent_text_np_leng_short), 'Patents',
                              np.array(wiki_np_leng_short), 'Wikipedia',
                              np.array(sem_abs_np_leng_short), 'Semantic Abstracts',
                              'Noun Phrase length',
                              'blue', 'green', 'red')

    plot_hist_2_distributions(np.array(patent_text_np_leng_short), 'Patents',
                              np.array(wiki_np_leng_short), 'Wikipedia', 'Noun Phrase length', 'blue', 'green')
    plot_hist_2_distributions(np.array(patent_text_np_leng_short), 'Patents',
                              np.array(sem_abs_np_leng_short), 'Semantic Scholar', 'Noun Phrase length', 'blue', 'red')


if __name__ == '__main__':
    patent_loc = sys.argv[1]
    sem_loc = sys.argv[2]
    wiki_loc = sys.argv[3]
    #patent_loc = 'data/patent/part-000000000674_preprocessed_wo_claims.pkl'
    #sem_loc = 'data/semanticscholar/s2-corpus-000_clean.pkl'
    #wiki_loc = 'data/wikitext/wikitext-2-raw/wiki.train.raw'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(patent_loc), str(sem_loc), str(wiki_loc))


