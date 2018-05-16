from projectFiles import utils
from projectFiles import matrixutils as mu

import numpy as np
from sklearn.cluster import KMeans

from matplotlib import pylab


path_to_input = '/home/paulojeunon/Desktop/TDP_NLP/books/materials_selection_in_mechanic_design/input/txtFiles/'
input_name = 'input.txt'
encoding = 'utf-8'

text_string_input = utils.read_text_input(path_to_input + input_name, encoding, True)
tokens_list = utils.tokenize_string(text_string_input, True)
tagged_tokens = utils.tag_tokens_using_stanford_corenlp(tokens_list)
# Will create context windows (word windows). A list that contains lists of tagged words
content_dict = utils.tokens_to_centralized_windows(tagged_tokens, 15, True)

windows = content_dict['windows']
# Will create the co-occurrence matrix with the obtained windows
cooc_matrix = mu.CoocMatrix(windows, enable_lemmatization=True)

U, s, Vh = np.linalg.svd(cooc_matrix.matrix.transpose(), full_matrices=True)

verb_pos_list = zip(U[:,0], U[:,1])
X = np.array(verb_pos_list)
model = KMeans(n_clusters=6).fit(X)



