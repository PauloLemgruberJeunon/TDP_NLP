import nltk
from nltk.tag.stanford import CoreNLPPOSTagger
import matrixutils as mu
import utils
from tkinter import *
from wordvectorsgui import vector_draw_gui

def main():
    workbook = utils.create_workbook('test-lemma.xlsx')
    worksheet = utils.get_new_worksheet('cooc_matrix_filtered.xlsx', workbook)
    worksheet2 = utils.get_new_worksheet('cooc_matrix_full.xlsx', workbook)

    file = open('pdfToTxt.txt', 'r', encoding="utf8")
    raw_text = file.read()
    raw_text = raw_text.lower()

    tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw_text)
    tokens = tokens[4:6000]

    tagger = CoreNLPPOSTagger(url='http://localhost:9000')

    tagged_text = tagger.tag(tokens)

    f2 = open('tagged_text.txt', 'w', encoding="utf8")
    for (word, tag) in tagged_text:
        f2.write(word + ' -> ' + tag + '\n')

    f2.close()

    windows = utils.tokens_to_windows(tagged_text, 14)

    cooc_matrix = mu.CoocMatrix(windows)

    cooc_matrix.filtered_matrix = mu.CoocMatrix.calc_ppmi(cooc_matrix.filtered_matrix, 1, 0.5)
    cooc_matrix.matrix = mu.CoocMatrix.calc_ppmi(cooc_matrix.matrix, 1, 0.5)

    utils.write_cooc_matrix(utils.invert_dictionary(cooc_matrix.filtered_noun_rows),
                            utils.invert_dictionary(cooc_matrix.filtered_verb_columns), cooc_matrix.filtered_matrix,
                            worksheet)

    utils.write_cooc_matrix(utils.invert_dictionary(cooc_matrix.noun_rows),
                            utils.invert_dictionary(cooc_matrix.verb_columns), cooc_matrix.matrix,
                            worksheet2)

    utils.close_workbook(workbook)

    root = Tk()
    vec_gui = vector_draw_gui(root, cooc_matrix)
    root.mainloop()

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
