import nltk
import numpy as np
from nltk.tag.stanford import CoreNLPPOSTagger
# import matplotlib.pyplot as plt
import heapq

raw = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
raw = raw.lower() #Makes the text

tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw)
tokens = tokens[30:6000]

tagger = CoreNLPPOSTagger(url='http://localhost:9000')

tagged_text = tagger.tag(tokens)

i=0
tagged_text_size = len(tagged_text)
temp_list = []
windows = []
while i < tagged_text_size:
    temp_list.append(tagged_text[i])
    if (i+1) % 14 == 0:
        windows.append(temp_list.copy())
        temp_list.clear()
    i+=1

if tagged_text_size % 14 != 0:
    windows.append(temp_list.copy())

noun_rows = {}
noun_rows_size = 0
verb_columns = {}
verb_columns_size = 0
cooc_matrix = np.array([[0,0],[0,0]])
window_verbs = []
window_nouns = []

for window in windows:
    for (word,tag) in window:
        if tag.startswith('V') == True:
            window_verbs.append(word)
            if word not in verb_columns:
                verb_columns[word] = verb_columns_size
                verb_columns_size+=1
                if(verb_columns_size > 2):
                    cooc_matrix = np.lib.pad(cooc_matrix, ((0,0),(0,1)), 'constant', constant_values=(0))

        if tag.startswith('NN') == True:
            window_nouns.append(word)
            if word not in noun_rows:
                noun_rows[word] = noun_rows_size
                noun_rows_size+=1
                if(noun_rows_size > 2):
                    cooc_matrix = np.lib.pad(cooc_matrix, ((0,1),(0,0)), 'constant', constant_values=(0))

    for verb in window_verbs:
        j = verb_columns.get(verb)
        for noun in window_nouns:
            i = noun_rows.get(noun)
            cooc_matrix[i][j] += 1

    window_verbs.clear()
    window_nouns.clear()

# plt.quiver(<list of x coordinates of origin>,<list of y coordinates of origin>,
#            <list of x coordinates of the end>,<list of y coordinates of the end>, color='<alguma cor>')

rows_cooc = np.zeros(noun_rows_size)
for i in range(0, noun_rows_size):
    rows_cooc[i] = np.sum(cooc_matrix[i])

thirty_percent = int(np.ceil(noun_rows_size*0.3))
thirty_percent_bigger_ind = heapq.nlargest(thirty_percent, range(noun_rows_size), rows_cooc.take)

chopped_cooc_matrix = np.empty((thirty_percent,verb_columns_size))

for i in range(0, thirty_percent):
    chopped_cooc_matrix[i] = cooc_matrix[thirty_percent_bigger_ind[i]].copy()

i = 0
temp_dict = dict(zip(noun_rows.values(), noun_rows.keys()))
noun_rows.clear()
while i < thirty_percent:
    noun_rows[temp_dict[thirty_percent_bigger_ind[i]]] = i
    i+=1


