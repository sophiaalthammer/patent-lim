# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import sys
import os
import errno
from definitions import ROOT_DIR


def year(publication_date: pd.Series):
    """
    Returns a numpy array containing the year for every row in the pd.Series
    :param publication_date: pd.Series where every row is in the format of yyyymmdd.xx
    :return: numpy array with the year as int
    """
    return np.array(publication_date // 10000).astype(int)


def get_subclass(tag_column: pd.Series):
    """
    returns an numpy array containing the main subclass per patent (the most frequent class of all ipc tags) of
    the ipc or cpc tag depending on the input column for all patents in the dataframe
    :param tag_column: pd.Series in the format of a string like 'F15B13/4022;...'
    :return: numpy array with the main subclass for every row in dataframe
    """
    list_tags = tag_column.str.split(';')
    for i in range(len(list_tags)):
        tags = []
        if type(list_tags[i]) == list:
            for j in range(len(list_tags[i])):
                tags.append(list_tags[i][j][:4])
            list_tags[i] = max(set(tags), key=tags.count)
    return np.array(list_tags)


def get_top_n_subclasses(tag_column: pd.Series, n: int):
    """
    returns an numpy array containing the top n main subclasses per patent (the most frequent class of all ipc tags) of
    the ipc or cpc tag depending on the input column for all patents in the dataframe
    :param tag_column: pd.Series in the format of a string like 'F15B13/4022;...'
    :param n: number of top subclasses which should be returned
    :return: numpy array with list of n top subclasses per row of dataframe
    """
    list_tags = tag_column.str.split(';')
    for i in range(len(list_tags)):
        tags = []
        if type(list_tags[i]) == list:
            for j in range(len(list_tags[i])):
                tags.append(list_tags[i][j][:4])
            # Get the unique tags sorted descending by frequency
            tags_sorted = sorted(set(tags), key=tags.count, reverse=True)
            # Get top n values from the sorted tags
            list_tags[i] = tags_sorted[:n]
    return np.array(list_tags)


def get_section(tag_column: pd.Series):
    """
    returns a numpy array containing the main section per patent (the most frequent section of all tags) of the ipc
    or cpc tag depending on the input column for all patents in the dataframe
    :param tag_column: pd.Series in the format of a string like 'F15B13/4022;...'
    :return: numpy array with the main section for every row in dataframe
    """
    list_tags = tag_column.str.split(';')
    for i in range(len(list_tags)):
        tags = []
        if type(list_tags[i]) == list:
            for j in range(len(list_tags[i])):
                tags.append(list_tags[i][j][0])
            list_tags[i] = max(set(tags), key=tags.count)
    return np.array(list_tags)


def get_class(tag_column: pd.Series):
    """
    returns a numpy array containing the main class per patent (the most frequent section of all tags) of the ipc
    or cpc tag depending on the input column for all patents in the dataframe
    :param tag_column: pd.Series in the format of a string like 'F15B13/4022;...'
    :return: numpy array with the main class for every row in dataframe
    """
    list_tags = tag_column.str.split(';')
    for i in range(len(list_tags)):
        tags = []
        if type(list_tags[i]) == list:
            for j in range(len(list_tags[i])):
                tags.append(list_tags[i][j][:3])
            list_tags[i] = max(set(tags), key=tags.count)
    return np.array(list_tags)


def get_all_subclasses(tag_column: pd.Series):
    """
    returns a numpy array containing all subclasses per patent (the most frequent section of all tags) of the ipc
    or cpc tag depending on the input column for all patents in the dataframe
    :param tag_column: pd.Series in the format of a string like 'F15B13/4022;...'
    :return: numpy array with all subclasses for every row in dataframe
    """
    list_tags = tag_column.str.split(';')
    for i in range(len(list_tags)):
        if type(list_tags[i]) == list:
            for j in range(len(list_tags[i])):
                list_tags[i][j] = list_tags[i][j][0:4]
    return np.array(list_tags)


def plot_hist(array: np.array, xaxis_title: str, title: str):
    """
    plots a histogram for the given numpy array
    :param array: numpy array containing numbers
    :return: shows a histogram which displays the frequency for each number (for example year)
    """
    plt.figure()
    plot = sns.distplot(array, bins="doane", hist_kws={"align": "left"}, color="orange")  # , kde=False
    plt.axvline(x=np.mean(array), color='orange', linestyle='--')
    plot.set(xticks=range(0, 70000, 10000))
    plot.set_xlim([-5000, 70000])
    #plot.set(xticks=range(0, 60, 10))
    #plot.set(xticks=range(0, math.ceil(max(array)/1000)*1000, 1000))
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel(xaxis_title)
    file_name = os.path.join(ROOT_DIR, 'plots/{0}_{1}_frequency.svg'.format(xaxis_title, title))
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    plt.savefig(file_name)
    plt.show()


def plot_tags(tags: np.array):
    """
    plots a barplot for the occurrences of all tags given in the numpy array
    :param tags: numpy array containing tags like section or class or subclass for each patent
    :return: plots and saves a barplot
    """
    df = pd.DataFrame.from_dict(Counter(tags), orient='index')
    df_ordered = df.sort_index()
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.4)
    #plt.figure(figsize=(7, 6))
    #fig, ax = plt.subplots()
    chart = sns.barplot(y=0, x=df_ordered.index, data=df_ordered) #, ax=ax, aspect=2)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=7)
    #ax.patches[1].set_width(100)
    #sns.axes_style("ticks", {"ytick.major.size": 8})
    plt.xlabel("CPC Sections")
    plt.ylabel('Frequency')
    file_name = os.path.join(ROOT_DIR, 'plots/{0}.svg'.format(tags[0]))
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    plt.savefig(file_name)
    plt.show()


def barplot_year_tags(years: np.array, tags: np.array, tag_name: str):
    """
    creates a stacked barplot with the years on x-axis and the bars subsectioned into the tags
    :param years: np.array containing the year for each patent
    :param tags: np.array containing the tag for each patent
    :param tag_name: name of the tag like 'section', 'class' or 'subclass'
    :return: plots and saves a figure
    """
    df = pd.DataFrame({'years': years, tag_name: tags})
    df2 = df.groupby(['years', tag_name]).size().reset_index()
    pivot_df = df2.pivot(index='years', columns=tag_name, values=0)
    pivot_df.plot.bar(stacked=True)
    plt.ylabel('frequency')
    file_name = os.path.join(ROOT_DIR,'plots/years_sections.svg')
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    plt.savefig(file_name)
    plt.show()


def main(ROOT_DIR, file_name):
    """
    Plots year of publication date, cpc and ipc subclass, class and section, top2 subclass occurences and years and
    sections
    :param ROOT_DIR: root directory of data folder
    :param file_name: file name of the csv file
    :return:
    """
    df = pd.read_csv(os.path.join(ROOT_DIR, file_name))

    # Plot years of publication date
    years = year(df['filing_date'])
    plot_hist(years, "Years", 'Years')

    # Plot cpc subclass occurences
    cpcs = get_subclass(df['cpc'])
    print(get_section(df['ipc']))
    plot_tags(cpcs)

    # Plot section occurences
    sections = get_section(df['cpc'])
    sections[sections == 'F'] = 'F - Mechanical Engineering'
    sections[sections == 'A'] = 'A - Human Necessities'
    sections[sections == 'B'] = 'B - Performing Operations'
    sections[sections == 'C'] = 'C - Chemistry'
    sections[sections == 'D'] = 'D - Textiles'
    sections[sections == 'E'] = 'E - Fixed Constructions'
    sections[sections == 'G'] = 'G - Physics'
    sections[sections == 'H'] = 'H - Electricity'
    sections[sections == 'Y'] = 'Y - Technological developments'
    plot_tags(sections)

    # Plot class occurences
    classes = get_class(df['cpc'])
    plot_tags(classes)

    # Print get top 2 subclasses for the df
    print(get_top_n_subclasses(df['cpc'], 2))

    # Plot years and sections
    barplot_year_tags(years, sections, 'Sections')
    tags = sections
    tag_name = 'Sections'
    df = pd.DataFrame({'years': years, tag_name: tags})
    df2 = df.groupby(['years',tag_name]).size().reset_index()
    pivot_df = df2.pivot(index='years', columns=tag_name, values=0)
    pivot_df.plot.bar(stacked=True, figsize=(10, 7))
    plt.show()
    df2.plot(kind='bar', stacked=True)
    plt.show()
    df.set_index(tag_name) \
       .reindex(df.set_index(tag_name).sum().sort_values().index, axis=1) \
       .T.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.show()


if __name__ == '__main__':
    file_name = sys.argv[1]
    # file_name = 'data/patent/part-000000000674'
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name))

