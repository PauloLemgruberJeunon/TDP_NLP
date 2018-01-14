import nltk
from nltk.tag.stanford import CoreNLPPOSTagger
# import matplotlib.pyplot as plt
import matrixutils as mu
import utils

def main():
    workbook = utils.create_workbook('test.xlsx')
    worksheet = utils.get_new_worksheet('cooc_matrix', workbook)


    utils.close_workbook(workbook)

    raw = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
    raw = raw.lower()  # Makes the text

    tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw)
    tokens = tokens[30:6000]

    tagger = CoreNLPPOSTagger(url='http://localhost:9000')

    tagged_text = tagger.tag(tokens)

    windows = utils.tokens_to_windows(tagged_text, 14)

    cooc_matrix = mu.CoocMatrix(windows)

    mu.CoocMatrix.calc_ppmi(cooc_matrix.filtered_matrix, 0)

    print(cooc_matrix.filtered_matrix)

    utils.write_cooc_matrix(cooc_matrix.filtered_noun_rows,cooc_matrix.filtered_verb_columns, cooc_matrix.filtered_matrix,
                            worksheet)

    quit()


if __name__ == "__main__":
    main()


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
