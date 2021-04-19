# 
# patent_lim: Linguistically informed masking for representation learning in the patent domain
#
# Copyright (c) Siemens AG, 2020
#
# SPDX-License-Identifier: Apache-2.0
#
import pandas as pd
import numpy as np
import os
import pickle
from definitions import ROOT_DIR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # All unique tags contained in all patents
    with open(os.path.join(ROOT_DIR, 'data/citations/unique_tags_all_documents.pkl'), "rb") as fp:
        unique_tags_all_doc = pickle.load(fp)
    # Positive citations
    with open(os.path.join(ROOT_DIR, "data/citations/cit_vec_sample100000_positive_augmented.pkl"), "rb") as fp:
        cit_vec = pickle.load(fp)
    # Negatives citations
    with open(os.path.join(ROOT_DIR, "data/citations/cit_vec_sample100000_negative_augmented.pkl"), "rb") as fp:
        cit_vec_neg = pickle.load(fp)

    X, y = get_train_test_data(cit_vec, cit_vec_neg, unique_tags_all_doc)
    print(X.shape)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train classifier
    #clf = SVC(kernel='rbf', C=100)
    clf = RandomForestClassifier(max_depth=100, n_estimators=1000)
    clf.fit(X_train, y_train)

    # Evaluate classifier
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def get_train_test_data(cit_vec, cit_vec_neg, unique_tags_all_doc):
    # Training and test data
    X_pos = get_x_data_from_citation_vectors(cit_vec, unique_tags_all_doc)
    y_pos = np.ones(X_pos.shape[0])

    X_neg = get_x_data_from_citation_vectors(cit_vec_neg, unique_tags_all_doc)
    y_neg = np.zeros(X_neg.shape[0])

    # Remove duplicates for negative samples which are contained in positive and negative samples
    df = pd.DataFrame(np.concatenate((X_pos, X_neg)))
    index = df.drop_duplicates(keep=False).index
    df = df.drop_duplicates(keep=False)
    y = np.concatenate((y_pos, y_neg))
    y = y[index]
    df['label'] = y
    indexes_to_drop = df[df['label'] == 1].index
    df.drop(indexes_to_drop, inplace=True)
    df = df.drop(columns=['label'])
    X_neg = df.to_numpy()

    # Merge positive and negative samples
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, np.zeros(X_neg.shape[0])))
    return X, y


def get_x_data_from_citation_vectors(cit_vec: list, unique_tags_all_doc: list):
    """
    Converts the data given in the cit_vectors into an np.array of the format (n_samples, n_features) where
    n_features = 2*len(unique_tags_all_doc)
    """
    mul_hot_vectors = []
    for j in range(len(cit_vec)):
        mul_hot_vec = [1 if i in cit_vec[j] else 0 for i in range(2*len(unique_tags_all_doc))]
        mul_hot_vectors.append(mul_hot_vec)
    return np.array(mul_hot_vectors)


if __name__ == '__main__':
    main()






