import nltk
import numpy as np
from nltk.tag.stanford import CoreNLPPOSTagger
# import matplotlib.pyplot as plt
import heapq
import xlsxwriter


def calcPPMI(cooc_matrix, improver, constant=0):
    """
    Wights the co-occurrence matrix using PPMI method and some selectable improover to this method

    Parameters
    ----------
    cooc_matrix : numpy matrix, the co-occurrence matrix
    improver : int, selects between none, Laplace smoothing and Palpha
    constant : int, can be the constant of Laplace or Palpha methods

    Returns
    -------
    None
    """
    # Laplace smoothing
    if improver is 1:
        np.add(cooc_matrix, constant)

    # Sum all the elements in the matrix
    total = cooc_matrix.sum()
    # Creates an array with each element containing the sum of one column
    total_per_column = cooc_matrix.sum(axis=0, dtype='int')
    # Creates an array with each element containing the sum of one row
    total_per_line = cooc_matrix.sum(axis=1, dtype='int')

    # Get the matrix dimensions
    (maxi, maxj) = cooc_matrix.shape

    # Iterates over all the matrix
    for i in range(maxi):
        for j in range(maxj):

            # Calculates the PMI's constants
            pcw = cooc_matrix[i][j]/total
            pw = total_per_line[i]/total
            pc = total_per_column[j]/total

            # Checks for division per zero
            if pw*pc == 0:
                cooc_matrix[i][j] = 0
            else:
                # Calculates the new wighted value
                cooc_matrix[i][j] = np.maximum(np.log2(pcw/(pw*pc)), 0)


workbook = xlsxwriter.Workbook('teste.xlsx')
worksheet = workbook.add_worksheet()

raw = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
raw = raw.lower()  # Makes the text

tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw)
tokens = tokens[30:6000]

tagger = CoreNLPPOSTagger(url='http://localhost:9000')

tagged_text = tagger.tag(tokens)

i = 0
tagged_text_size = len(tagged_text)
temp_list = []
windows = []
while i < tagged_text_size:
    temp_list.append(tagged_text[i])
    if (i + 1) % 14 == 0:
        windows.append(temp_list.copy())
        temp_list.clear()
    i += 1

if tagged_text_size % 14 != 0:
    windows.append(temp_list.copy())

noun_rows = {}
noun_rows_size = 0
verb_columns = {}
verb_columns_size = 0
cooc_matrix = np.array([[0, 0], [0, 0]])
window_verbs = []
window_nouns = []

for window in windows:
    for (word, tag) in window:
        if tag.startswith('V') is True:
            window_verbs.append(word)
            if word not in verb_columns:
                verb_columns[word] = verb_columns_size
                verb_columns_size += 1
                if verb_columns_size > 2:
                    cooc_matrix = np.lib.pad(cooc_matrix, ((0, 0), (0, 1)), 'constant', constant_values=0)

        if tag.startswith('NN') is True:
            window_nouns.append(word)
            if word not in noun_rows:
                noun_rows[word] = noun_rows_size
                noun_rows_size += 1
                if noun_rows_size > 2:
                    cooc_matrix = np.lib.pad(cooc_matrix, ((0, 1), (0, 0)), 'constant', constant_values=0)

    for verb in window_verbs:
        j = verb_columns.get(verb)
        for noun in window_nouns:
            i = noun_rows.get(noun)
            cooc_matrix[i][j] += 1

    window_verbs.clear()
    window_nouns.clear()

# plt.quiver(<list of x coordinates of origin>,<list of y coordinates of origin>,
#            <list of x coordinates of the end>,<list of y coordinates of the end>, color='<alguma cor>')


thirty_percent = int(np.ceil(noun_rows_size * verb_columns_size * 0.3))
temp_matrix_list = np.empty(noun_rows_size * verb_columns_size)
x = 0
for i in cooc_matrix.flat:
    temp_matrix_list[x] = i
    x += 1

thirty_percent_bigger_ind = heapq.nlargest(thirty_percent, range(noun_rows_size * verb_columns_size),
                                           temp_matrix_list.take)

real_aij_matrix_pos = []
while i < thirty_percent:
    real_aij_matrix_pos.append([thirty_percent_bigger_ind[i] // verb_columns_size,
                               thirty_percent_bigger_ind[i] % verb_columns_size])
    i += 1

new_verb_columns_size = 0
new_noun_rows_size = 0
noun_row_idxs = {}
verb_column_idxs = {}
for two_index in real_aij_matrix_pos:
    if two_index[0] not in noun_row_idxs:
        noun_row_idxs[two_index[0]] = 0
        new_noun_rows_size += 1
    if two_index[1] not in verb_column_idxs:
        verb_column_idxs[two_index[1]] = 0
        new_verb_columns_size += 1

new_verb_columns = {}
new_noun_rows = {}
k = l = 0
new_cooc_matrix = np.zeros((new_noun_rows_size, new_verb_columns_size))

temp_dict_noun = dict(zip(noun_rows.values(), noun_rows.keys()))
temp_dict_verb = dict(zip(verb_columns.values(), verb_columns.keys()))

for i in range(noun_rows_size):
    if i not in noun_row_idxs:
        continue
    else:
        new_noun_rows[temp_dict_noun[i]] = k
        l = 0
    for j in range(verb_columns_size):
        if j not in verb_column_idxs:
            continue
        else:
            new_cooc_matrix[k][l] = cooc_matrix[i][j]
            new_verb_columns[temp_dict_verb[j]] = l
            l += 1
    k += 1

calcPPMI(new_cooc_matrix)

for i in range(noun_rows_size):
    worksheet.write(i+1, 0, temp_dict_noun[i])

for j in range(verb_columns_size):
    worksheet.write(0, j+1, temp_dict_verb[j])

for i in range(noun_rows_size):
    for j in range(verb_columns_size):
        worksheet.write(i + 1, j + 1, cooc_matrix[i][j])

workbook.close()

# print(new_noun_rows_size)
# print(noun_rows_size)
# print('')
# print(new_verb_columns_size)
# print(verb_columns_size)

# print(real_aij_matrix_pos)

# 30% bigger in module vectors (not what was asked)
# rows_cooc = np.zeros(noun_rows_size)
# for i in range(0, noun_rows_size):
#     rows_cooc[i] = np.sum(cooc_matrix[i])
#
# thirty_percent = int(np.ceil(noun_rows_size*0.3))
# thirty_percent_bigger_ind = heapq.nlargest(thirty_percent, range(noun_rows_size), rows_cooc.take)
#
# chopped_cooc_matrix = np.empty((thirty_percent,verb_columns_size))
#
# for i in range(0, thirty_percent):
#     chopped_cooc_matrix[i] = cooc_matrix[thirty_percent_bigger_ind[i]].copy()
#
# i = 0
# temp_dict = dict(zip(noun_rows.values(), noun_rows.keys()))
# noun_rows.clear()
# while i < thirty_percent:
#     noun_rows[temp_dict[thirty_percent_bigger_ind[i]]] = i
#     i+=1
