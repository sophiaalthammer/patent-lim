import spacy
import os
import sys
from definitions import ROOT_DIR
import numpy as np


def get_mapping():
    """
    Returns a dictionary with the mapping of Spacy dependency labels to a numeric value, spacy dependency annotations
    can be found here https://spacy.io/api/annotation
    :return: dictionary
    """
    keys = ['acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp',
            'clf', 'compound', 'conj', 'cop', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'discourse', 'dislocated',
            'dobj', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'intj', 'list', 'mark',
            'meta', 'neg', 'nn', 'nmod', 'nounmod', 'npadvmod', 'npmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'obj', 'obl',
            'orphan', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl',
            'reparandum', 'root', 'vocative', 'xcomp', '']
    values = list(range(2, len(keys) + 2))
    assert len(keys) == len(values)
    return dict(zip(keys, values))


def get_noun_phrase_vector(nlp, string: str, mapping: dict):
    """
    Calculates the vector denoting for every token in a string if the token belongs to a noun phrase (value of token>=1)
    and then also which role the token has in the noun phrase. Head of the noun phrase is denoted by 1, all other tokens
    in noun phrase have number depending on their dependency structure to the head noun. The number is defined by the
    mapping of the dependency labels in spacy mapped to a number
    :param nlp: spacy language model
    :param string: string text to which the noun phrase vector should be calculated
    :param mapping: dictionary with mapping of spacy dependency labels to a number
    :return: vector of the length of the tokens in string
    """
    doc = nlp(string)
    positions_of_np = [token.i for chunk in doc.noun_chunks for token in chunk]
    positions_chunk_root = [chunk.root.i for chunk in doc.noun_chunks]
    np_vector = [0] * len(doc)
    #np_vector = np.zeros(len(doc))
    np_vector = [1 if i in positions_chunk_root else 0 for i in range(len(doc))]
    #np_vector[i for i in positions_chunk_root] = 1
    for i in set(positions_of_np)-set(positions_chunk_root):
        np_vector[i] = mapping[doc[i].dep_]
    return np_vector


def main(input_file: str, output_file: str):
    """
    Writes for each line in the input_file location the noun phrase vector to the output file location, for empty lines
    in the input file it also writes empty lines in the output file
    :param input_file: str input file location
    :param output_file: str output file location
    :return:
    """
    # wichtig damit die array2string nicht abgeschnitten werden durch line breaks sondern ein array pro line ist
    np.set_printoptions(linewidth=np.nan)
    mapping = get_mapping()
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1500000
    with open(os.path.join(ROOT_DIR, input_file), 'r') as f:
        with open(os.path.join(ROOT_DIR, output_file), 'w') as g:
            for line in f.readlines():
                if line == '\n':
                    g.write('\n')
                else:
                    g.write('[' + ', '.join(str(x) for x in get_noun_phrase_vector(nlp, line.strip(), mapping)) + ']' + '\n')


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # input_file = 'bert/data/npvector_test_text.txt'
    # output_file = 'bert/data/npvector_test_vectors.txt'
    main(str(input_file), str(output_file))







