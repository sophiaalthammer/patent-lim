import pandas as pd
import numpy as np
import os
import sys
import gc
import pickle


def compute_average_subclass_number(file_location: str, file, new_file_location: str):
    file_name = 'part-000000000{}.csv.gz'.format(file)
    df = pd.read_csv(os.path.join(file_location, file_name), compression='gzip', error_bad_lines=False)

    # Drop columns which we dont need
    df = df.drop(['publication_date', 'filing_date', 'priority_date',
                  'title_truncated', 'abstract_truncated', 'claim_truncated', 'description_truncated'], axis=1)

    # Remove rows with empty or NaN cells
    df.replace("", float("NaN"), inplace=True)
    df.dropna(subset=['title', 'abstract', 'claim', 'description', 'ipc', 'cpc'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create new columns with top1 of IPC and CPC tag
    df['ipc_number'] = get_subclass_number(df['ipc'])
    df['cpc_number'] = get_subclass_number(df['cpc'])

    assert df.loc[df['ipc_number'].isnull()].empty
    assert df.loc[df['cpc_number'].isnull()].empty

    ipc_average = df['ipc_number'].mean()
    cpc_average = df['cpc_number'].mean()

    print('Average number of IPC subclasses: {0}'.format(ipc_average))
    print('Average number of CPC subclasses: {0}'.format(cpc_average))

    # Store the data in the tsv file which will be used for training plus the unique tags contained in the file
    #with open(os.path.join(new_file_location, 'ipc_number_unique_tags_0{0}.pkl'.format(file)), "wb") as fp:
    #    pickle.dump(list(set(df['ipc_number'])), fp)
    #with open(os.path.join(new_file_location, 'cpc_number_unique_tags_0{0}.pkl'.format(file)), "wb") as fp:
    #    pickle.dump(list(set(df['cpc_number'])), fp)

    #df.to_csv(os.path.join(new_file_location, '{0}_0{1}_ipc_cpc_number.tsv'.format(train_test, file)), sep='\t', index=False,
    #          header=False)

    #del [df]
    #gc.collect()
    return df


def get_subclass_number(tag_column: pd.Series):
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
            list_tags[i] = len(set(tags))
    return np.array(list_tags)


if __name__ == '__main__':
    #file_location = sys.argv[1]
    #new_file_location = sys.argv[2]
    #train_test = sys.argv[3]
    #file = sys.argv[4]
    #start_files = sys.argv[4]
    #end_files = sys.argv[5]
    file_location = '/home/ubuntu/Documents/thesis/data/patent_contents_en_since_2000_application_kind_a'
    new_file_location = '/home/ubuntu/PycharmProjects/patent/data/ipc_classification'
    train_test = 'train'
    #start_files = 0
    #end_files = 3
    file = 191
    #main(str(file_location), str(new_file_location), str(train_test), int(start_files), int(end_files))
    df = compute_average_subclass_number(str(file_location), "{0:03}".format(int(file)), str(new_file_location))

    # files = ["{0:03}".format(i) for i in range(3)]
    # merge_unique_tags_all_documents(new_file_location, files, 'ipc')
    # merge_unique_tags_all_documents(new_file_location, files, 'cpc')
    # #This is how I could join them later for the classification when reading them in in the DataProcessor
    # print(df['title'].iloc[0] + '. ' + df['abstract'].iloc[0])

    # Store the data in the tsv file which will be used for training plus the unique tags contained in the file
    #with open(os.path.join(new_file_location, 'cpc_unique_tags_0{0}.pkl'.format(file)), "wb") as fp:
    #    pickle.dump(list(set(df[8])), fp)
