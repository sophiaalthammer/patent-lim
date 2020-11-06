import pandas as pd
import pickle
import os
import sys
from definitions import ROOT_DIR


def get_claims_documents(index: list, file_name: str):
    """
    Gets the unique tags for all documents contained in the list index and saves them as a pickle file
    """
    for i in index:
        df_location = '{0}{1}'.format(file_name.split('.')[0][:-len(i)], i)
        df = pd.read_csv(os.path.join(ROOT_DIR, '{0}.csv.gz'.format(df_location)), compression='gzip', error_bad_lines=False)
        df = df.drop(['pub_date', 'filing_date', 'priority_date', 'title', 'abstract', 'ipc'], axis=1)
        df['claim'] = df['claim'].str.replace('[ \t\n]+', ' ', regex=True)
        with open(os.path.join(ROOT_DIR, '{0}_wclaims.pkl'.format(df_location)), "wb") as fp:
            pickle.dump(df, fp)


def concat_claims_all_documents(index: list, file_name: str):
    """
    Merges all the lists of unique tags and saves the merged version as one pickle file with all unique tags
    """
    list_of_dfs = []
    for i in index:
        location = '{0}{1}_wclaims.pkl'.format(file_name.split('.')[0][:-len(i)], i)
        with open(os.path.join(ROOT_DIR, location), "rb") as fp:
            df_claims = pickle.load(fp)
            list_of_dfs.append(df_claims)
    df_concat_all_claims = pd.concat(list_of_dfs)
    df_concat_all_claims = df_concat_all_claims.reset_index(drop=True)
    with open(os.path.join(ROOT_DIR, '{0}_wclaims_all.pkl'.format(file_name.split('.')[0])), "wb") as fp:
        pickle.dump(df_concat_all_claims, fp)
    return df_concat_all_claims


def merge_pos_and_neg_citation_pairs(file_name: str):
    cit = pd.read_csv(os.path.join(ROOT_DIR,"{0}/citations-only-type-x-with_claims.csv".format(os.path.dirname(
        file_name))))
    cit_neg = pd.read_pickle(os.path.join(ROOT_DIR,"{0}/citations-only-type-x-with_claims_negatives.pkl".format(
        os.path.dirname(file_name))))

    cit['label'] = 1
    cit_neg['label'] = 0

    citations = pd.concat([cit, cit_neg])
    citations = citations.drop_duplicates(keep='first', subset=['pub_number', 'cit_number'])

    with open(os.path.join(ROOT_DIR, '{0}/citations-only-type-x-with_claims_posandneg.pkl'.format(
            os.path.dirname(file_name))), "wb") as fp:
        pickle.dump(citations, fp)


def append_claims_to_cit(claims, cit):
    claims = claims.rename(columns={"claim": "pub_claim"})
    citing_patents = pd.merge(cit, claims, on='pub_number', how='inner')
    # print(citing_patents[citing_patents['pub_claim'].isnull()==False].index)

    claims = claims.rename(columns={"pub_claim": "cit_claim"})
    citing_patents = pd.merge(citing_patents, claims, left_on='cit_number', right_on='pub_number', how='inner')

    citing_patents = citing_patents.drop(['pub_number_y'], axis=1)
    citing_patents = citing_patents.rename(columns={'pub_number_x': 'pub_number'})
    citing_patents.drop_duplicates(inplace=True)
    return citing_patents


def main(ROOT_DIR: str, file_name, start_index: int, end_index: int):
    index = ["{0:02}".format(i) for i in range(start_index, end_index)]
    # Save the claim of the documents
    get_claims_documents(index, file_name)

    # concat all claims from all separate files
    df = concat_claims_all_documents(index, file_name)
    # merge positive and negative citation pairs with labels
    merge_pos_and_neg_citation_pairs(file_name)
    # read them in
    cit = pd.read_pickle(os.path.join(ROOT_DIR,"{0}/citations-only-type-x-with_claims_posandneg.pkl".format(
        os.path.dirname(file_name))))

    # open all citation pairs with claims
    df = pd.read_pickle(os.path.join(ROOT_DIR, "{0}_wclaims_all.pkl".format(file_name.split('.'))))

    # merge labels to dataframe with citation pairs
    cit = append_claims_to_cit(df, cit)

    print('Number of negative citation pairs: {0}'.format(len(cit[cit['label'] == 0])))
    print('Number of positive citation pairs: {0}'.format(len(cit[cit['label'] == 1])))

    assert cit[cit['pub_claim'].isnull() == True].index is not None
    assert cit[cit['cit_claim'].isnull() == True].index is not None

    cit = cit.sample(frac=1).reset_index(drop=True)

    # take first 130000 citation pairs
    cit_train = cit[:130000]

    print('Number of negative citation pairs in 130k training citation pairs: {0}'.format(
        len(cit_train[cit_train['label'] == 0])))
    print('Number of positive citation pairs in 130k training citation pairs: {0}'.format(
        len(cit_train[cit_train['label'] == 1])))

    # save training data
    with open(os.path.join(ROOT_DIR,'{0}_train_data.pkl'.format(file_name.split('.')[0])), "wb") as fp:
        pickle.dump(cit_train, fp)

    cit_train.to_csv(os.path.join(ROOT_DIR,'{0}_train_data.tsv'.format(file_name.split('.')[0])),
                     sep='\t', index=False, header=False)

    # Save evaluation data
    cit_eval = cit[130000:146500]

    print('Number of negative citation pairs in 16.5k evaluation citation pairs: {0}'.format(
        len(cit_eval[cit_eval['label'] == 0])))
    print('Number of positive citation pairs in 16.5k evaluation citation pairs: {0}'.format(
        len(cit_eval[cit_eval['label'] == 1])))

    with open(os.path.join(ROOT_DIR,'{0}_eval_data.pkl'.format(file_name.split('.')[0])), "wb") as fp:
        pickle.dump(cit_eval, fp)

    cit_eval.to_csv(os.path.join(ROOT_DIR,'{0}_eval_data.tsv'.format(file_name.split('.')[0])),
                    sep='\t', index=False, header=False)


if __name__ == '__main__':
    file_name = sys.argv[1]
    start_index = sys.argv[2]
    end_index = sys.argv[3]
    # file_name = 'data/citations/patent-contents-for-citations-wclaims/patent-contents-for-citations_en_claim_wclaims000000000000.csv.gz'
    # start_index = 0
    # end_index = 10
    # ROOT_DIR = os.path.dirname(os.path.abspath('/home/ubuntu/PycharmProjects/patent/requirements.txt'))
    main(ROOT_DIR, str(file_name), int(start_index), int(end_index))






