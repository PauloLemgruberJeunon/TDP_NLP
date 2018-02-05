import nltk
from nltk.tag.stanford import CoreNLPPOSTagger
import matrixutils as mu
import utils
from tkinter import *
from wordvectorsgui import VectorDrawGui


load_matrix = False


def main():

    if load_matrix is not True:
        workbook = utils.create_workbook('test-lemma-42.xlsx')
        worksheet = utils.get_new_worksheet('cooc_matrix_filtered', workbook)
        worksheet2 = utils.get_new_worksheet('cooc_matrix_full', workbook)
        worksheet3 = utils.get_new_worksheet('soc_pmi_matrix', workbook)
        test_worksheet = utils.get_new_worksheet('test', workbook)

        tagger = CoreNLPPOSTagger(url='http://localhost:9000')

        file = open('pdfToTxt.txt', 'r', encoding="utf8")
        raw_text = file.read()
        raw_text = raw_text.lower()

        tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw_text)

        tagged_text = []
        txt_size = len(tokens)
        i = 0
        while i < txt_size:
            tokens_to_tag = tokens[i:i+6000]
            i += 6001
            if i+6000 >= txt_size:
                tokens_to_tag = tokens[i:txt_size]
                i = txt_size+1

            tagged_text += tagger.tag(tokens_to_tag)

        if input('Want to save tagged text? (y/n)').lower() == 'y':
            utils.save_tagged_words(tagged_text)

        windows = utils.tokens_to_centralized_windows(tagged_text, 15)

        cooc_matrix = mu.CoocMatrix(windows)

        utils.write_cooc_matrix(utils.invert_dictionary(cooc_matrix.filtered_noun_rows),
                                utils.invert_dictionary(cooc_matrix.filtered_verb_columns), cooc_matrix.filtered_matrix,
                                test_worksheet)

        cooc_matrix.filtered_matrix = mu.CoocMatrix.calc_ppmi(cooc_matrix.filtered_matrix, 1, 0.5)
        cooc_matrix.matrix = mu.CoocMatrix.calc_ppmi(cooc_matrix.matrix, 1, 0.5)

        cooc_matrix.is_pmi_calculated = True

        cooc_matrix.create_soc_pmi_matrix(cooc_matrix.filtered_matrix)

        inverted_filtered_noun_dict = utils.invert_dictionary(cooc_matrix.filtered_noun_rows)

        utils.write_cooc_matrix(inverted_filtered_noun_dict,
                                utils.invert_dictionary(cooc_matrix.filtered_verb_columns), cooc_matrix.filtered_matrix,
                                worksheet)

        utils.write_cooc_matrix(utils.invert_dictionary(cooc_matrix.noun_rows),
                                utils.invert_dictionary(cooc_matrix.verb_columns), cooc_matrix.matrix,
                                worksheet2)

        utils.write_cooc_matrix(inverted_filtered_noun_dict,
                                inverted_filtered_noun_dict, cooc_matrix.soc_pmi_matrix,
                                worksheet3)

        utils.close_workbook(workbook)

    else:
        content = utils.load_from_wb('test-lemma.xlsx')
        cooc_matrix = mu.CoocMatrix(build_matrix=False, content=content)

    root = Tk()
    vec_gui = VectorDrawGui(root, cooc_matrix)
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
