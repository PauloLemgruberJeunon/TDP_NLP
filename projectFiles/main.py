import nltk
from nltk.tag.stanford import CoreNLPPOSTagger
import matrixutils as mu
import utils
from tkinter import *
from wordvectorsgui import VectorDrawGui

# Global variable just to decide if the cooc_matrix will be generated or loaded from the xlsx file
load_matrix = True


def main():

    # Here the matrix will be generated through the .txt file provided
    if load_matrix is not True:

        # The worksheet will be saved in a excel workbook with the file name and location equal to the string below
        workbook = utils.create_workbook('..\\xlsxFiles\\test-lemma-42.xlsx')

        # In the lines below the worksheets will be created and associated to the workbook
        worksheet = utils.get_new_worksheet('cooc_matrix_filtered', workbook)
        worksheet2 = utils.get_new_worksheet('cooc_matrix_full', workbook)
        worksheet3 = utils.get_new_worksheet('soc_pmi_matrix', workbook)
        test_worksheet = utils.get_new_worksheet('test', workbook)

        # This will attempt to connect to the local stanford coreNLP Server that must be running before the execution
        # of this code
        tagger = CoreNLPPOSTagger(url='http://localhost:9000')

        # This will open the text .txt archive and read it
        file = open('..\\txtFiles\\filtered_txt_output.txt', 'r', encoding="utf8")

        # This will transform the archive into a huge string
        raw_text = file.read()

        # This will lower all the upper case letters in the string (in the entire input file text)
        raw_text = raw_text.lower()

        # This will create a python list named 'tokens' that will have each word/number as an element
        # It will eliminate all kinds of non-alpha numerical characters (punctuation included)
        tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw_text)

        # The piece of code below is exists to deal with a limitation of the Stanford's coreNLP Server that only
        # supports 100000 characters per server call. So this will break the text in a lot of smaller pieces and send
        # them to the server and after will unite them all in one list of tagged words ('tagged_text')
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

        # Just a prompt to check if the user wants to save the tagged_text in a .txt file for further analysis
        if input('Want to save tagged text? (y/n)').lower() == 'y':
            utils.save_tagged_words(tagged_text)

        # Will create context windows (word windows). A list that contains lists of tagged words
        windows = utils.tokens_to_centralized_windows(tagged_text, 15)

        # Will create the co-occurrence matrix with the obtained windows
        cooc_matrix = mu.CoocMatrix(windows)

        # Just a test worksheet with the pure frequency numbers of the pairs of verbs and nouns
        utils.write_cooc_matrix(utils.invert_dictionary(cooc_matrix.filtered_noun_rows),
                                utils.invert_dictionary(cooc_matrix.filtered_verb_columns), cooc_matrix.filtered_matrix,
                                test_worksheet)

        # Will calculate the PPMI of the normal and the filtered matrices
        cooc_matrix.filtered_matrix = mu.CoocMatrix.calc_ppmi(cooc_matrix.filtered_matrix, 1, 0.5)
        cooc_matrix.matrix = mu.CoocMatrix.calc_ppmi(cooc_matrix.matrix, 1, 0.5)

        # This will set a variable of the object 'cooc_matrix' aloowing for the creation of the soc_pmi_matrix
        cooc_matrix.is_pmi_calculated = True

        # Will create the Second order co-occurrence matrix from the filtered co-occurrence matrix matrix
        cooc_matrix.create_soc_pmi_matrix(cooc_matrix.filtered_matrix)

        # Will invert the dictionary to aid the process of storing the co-occurrence matrices in xlsx files
        inverted_filtered_noun_dict = utils.invert_dictionary(cooc_matrix.filtered_noun_rows)

        # The worksheets are been filled and will be saved on the created workbook
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

    # In this case the co-occurrence will be loaded from the xlsx archives
    else:
        content = utils.load_from_wb('..\\xlsxFiles\\test-lemma.xlsx')
        cooc_matrix = mu.CoocMatrix(build_matrix=False, content=content)

    # Creation of the GUI
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
