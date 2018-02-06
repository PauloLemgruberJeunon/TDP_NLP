"""
This file intends to be an utility box, containing functions to help with smaller functionalities
"""

import xlsxwriter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from openpyxl import load_workbook
from openpyxl.utils import coordinate_from_string, column_index_from_string
import nltk


# Global dictionary used to quickly check if a verb in the text is one that we want to keep in the co-occurrence matrix.
# The values do not matter, only the keys
verbs_to_keep = {'prepare': 1, 'synthesize': 1, 'generate': 1, 'define': 1, 'illustrate': 1, 'classify': 1,
                'develop': 1, 'name': 1, 'defend': 1, 'explain': 1, 'describe': 1, 'criticize': 1,
                'test': 1, 'review': 1, 'order': 1, 'analyze': 1, 'choose': 1, 'create': 1, 'combine': 1, 'infer': 1,
                'extend': 1, 'modify': 1, 'compare': 1, 'indicate': 1, 'distinguish': 1, 'interpret': 1, 'justify': 1,
                'identify': 1, 'list': 1, 'evaluate': 1, 'calculate': 1, 'design': 1, 'recognize': 1, 'model': 1,
                'discuss': 1, 'practice': 1, 'apply': 1, 'estimate': 1, 'compute': 1, 'solve': 1, 'conclude': 1,
                'predict': 1}

path_to_xlsxFolder = '..\\xlsxFiles\\'
path_to_txtFolder = '..\\txtFiles\\'


def measure_postag_accuracy(tagged_sents, tagged_words):
    """
    Function used to measure the accuracy of a POSTagger (does not handle different types of tag standards)
    Parameters
    ----------
    tagged_sents : Are the sentences of the database used to measure the accuracy of the tagger ("Test set")
    tagged_words : Are the words of the sentences stored in "tagged_sents" the were tagged by the POSTagger

    Returns
    -------
    The measured accuracy of the POSTagger
    """

    total_count = 0
    right_count = 0

    for i in range(0, len(tagged_sents)):
        print('[Accuracy] Current stage = ' + str(i))
        k = 0
        j = 0
        local_count = 0
        local_right_count = 0
        while j < len(tagged_sents[i]):
            total_count += 1
            local_count += 1
            if tagged_sents[i][j][1] == '-NONE-':
                while tagged_sents[i][j][1] == '-NONE-':
                    j += 1
                while tagged_sents[i][j][0] != tagged_words[i][k][0]:
                    k += 1
            if tagged_sents[i][j][1] == tagged_words[i][k][1]:
                right_count += 1
                local_right_count += 1

            k += 1
            j += 1
        print('Local iter accuracy = ' + str(local_right_count/local_count) + '\n')

    return right_count/total_count


def create_workbook(name):
    #  Creates and return a Workbook with a chosen "name"
    return xlsxwriter.Workbook(name)


def get_new_worksheet(name, workbook):
    #  Add a new worksheet to a "workbook" and return it (returns the worksheet)
    return workbook.add_worksheet(name)


def close_workbook(workbook):
    #  Closes the workbook (mandatory by the end of its use)
    workbook.close()


def write_cooc_matrix(row_index_name_dict, column_index_name_dict, cooc_matrix, worksheet):
    """
    This function will print the matrix in a excel spreadsheet
    :param row_index_name_dict: The dictionary that stores the name of the rows (nouns) to print them in the sheet
    :param column_index_name_dict: The dictionary that stores the name of the columns (verbs) to print them in the sheet
    :param cooc_matrix: The co-occurrence matrix that will be written to the arquive
    :param worksheet: The worksheet that the matrix will be written to
    :return: Nothing
    """

    row_len = len(row_index_name_dict)
    column_len = len(column_index_name_dict)

    for i in range(row_len):
        worksheet.write(i + 1, 0, row_index_name_dict[i])

    for j in range(column_len):
        worksheet.write(0, j + 1, column_index_name_dict[j])

    for i in range(row_len):
        for j in range(column_len):
            worksheet.write(i + 1, j + 1, cooc_matrix[i][j])


def tokens_to_windows(tokens, window_size):
    """
    Transform a tokenized text into a series of windows for them to serve as contexts
    :param tokens: A python list containing the tokenized text
    :param window_size: The wanted window size
    :return: A python list containing the "tokens" separated in windows
    """
    i = 0
    tagged_text_size = len(tokens)
    temp_list = []
    windows = []
    while i < tagged_text_size:
        temp_list.append(tokens[i])
        if (i + 1) % window_size == 0:
            windows.append(temp_list.copy())
            temp_list.clear()
        i += 1

    if tagged_text_size % window_size != 0:
        windows.append(temp_list.copy())

    return windows


def tokens_to_centralized_windows(tagged_text, window_size):
    """
    This function is other windowing method. It will find the desired verbs and make windows around them. The windows
    can overlap, its normal.
    :param tagged_text: A list of tuples. The contents of the tuples are two string, the first being the word and the
    second being the tag related to the word
    :param window_size: It is an integer that indicates the wanted size of the windows
    :return: Returns a list of lists. Each inner list is a window containing the word-tag tuples
    """
    tagged_text_size = len(tagged_text)
    windows = []

    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Checks if the 'window_size' value is even or odd
    is_even = False
    if window_size % 2 == 0:
        is_even = True

    # Based of the window size being even or odd the code below calculates the right and left offsets from the central
    # element of the window, that is the wanted verbs in this case
    begin_offset = window_size // 2
    if is_even:
        end_offset = (window_size // 2) - 1
    else:
        end_offset = (window_size // 2)

    i = 0
    # Iterates over the tagged_text and create the windows based on it
    while i < tagged_text_size:

        # Checks if the current word is a verb and checks if it's infinitive form is present on the dict of wanted verbs
        if tagged_text[i][1].startswith('V') and lemmatizer.lemmatize(tagged_text[i][0], 'v') in verbs_to_keep:
            # Dumb but functional method to prevent windows from exceed the 'tagged-text' array's boundaries
            if i - begin_offset < 0 or i + end_offset >= tagged_text_size:
                if i - begin_offset < 0:
                    start = 0
                    end = i + end_offset
                else:
                    start = i - begin_offset
                    end = tagged_text_size - 1
            # The else holds the code for a regular execution
            else:
                start = i - begin_offset
                end = i + end_offset

            # Adds a window to the windows variables
            temp_list = tagged_text[start:end]
            windows.append(temp_list.copy())

        i += 1

    return windows


def invert_dictionary(dictionary):
    return dict(zip(dictionary.values(), dictionary.keys()))


def plot_vectors(vec1_coord, vec2_coord, vec1_name, vec2_name, verb1_name, verb2_name):
    # Pyplot method to clear the old drawings
    plt.gcf().clear()

    # Calculate the module of the arrays
    vec1_module = np.sqrt(np.power(vec1_coord[0], 2) + np.power(vec1_coord[1], 2))
    vec2_module = np.sqrt(np.power(vec2_coord[0], 2) + (np.power(vec2_coord[1], 2)))

    # Pyplot quiver plot to show the word vectors
    plt.quiver([0, 0], [0, 0], [vec1_coord[0]/vec1_module, vec2_coord[0]/vec2_module],
               [vec1_coord[1]/vec1_module, vec2_coord[1]/vec2_module], color=['r', 'g'], angles='xy',
               scale_units='xy', scale=1)

    # Calculates the axis limits (x_begin, x_end, y_begin, y_end)
    plt.axis([0, 1, 0, 1])

    # The plotted vector's legends
    lgd_red = mpatches.Patch(color='red', label=vec1_name)
    lgd_green = mpatches.Patch(color='green', label=vec2_name)

    # Adding the legends to the plot handler
    plt.legend(handles=[lgd_red, lgd_green])

    # Adding labels to the axis
    plt.xlabel(verb1_name)
    plt.ylabel(verb2_name)

    # Shows the plot in a window but does not blocks this thread (the 'False' parameter is for the blocking option)
    plt.show(False)


def get_column_names(rows, verb_columns):
    counter = 0
    for row in rows:
        for cell in row:
            value = cell.value
            if value is not None:
                verb_columns[value] = counter
                counter += 1
        break


def complete_the_loading(rows, noun_rows, matrix):
    counter = 0
    i = 0
    for row in rows:
        row_jump = True
        j = 0

        for cell in row:
            value = cell.value
            if row_jump:
                noun_rows[value] = counter
                counter += 1
                row_jump = False
            else:
                matrix[i][j] = value
                j += 1

        i += 1


def load_from_wb(workbook_name):
    """
    Function used to load the matrix data from a xlsx archive. It is faster then generating it from text
    :param workbook_name: path and name of the xlsx file to use to extract information
    :return: Returns a dictionary containing the information to create a 'CoocMatrix' object
    """
    # Load workbook
    wb = load_workbook(filename=workbook_name, read_only=True)
    print('wb loaded')

    # Load worksheets
    ws = wb['cooc_matrix_full']
    ws_filtered = wb['cooc_matrix_filtered']
    ws_soc_pmi = wb['soc_pmi_matrix']

    # Calculate the matrix dimensions and transform the excel's coordinates to matrix coordinates (to only integers)
    rows = ws.rows
    matrix_dim = ws.calculate_dimension().split(':')
    rows_count = coordinate_from_string(matrix_dim[1])[1] - 1
    column_count = column_index_from_string(coordinate_from_string(matrix_dim[1])[0]) - 1

    matrix = np.empty((rows_count, column_count))

    print('matrix allocated')

    noun_rows = {}
    verb_columns = {}

    get_column_names(rows, verb_columns)
    print('get_column_names completed')
    complete_the_loading(rows, noun_rows, matrix)
    print('complete_the_loading completed')

    rows = ws_filtered.rows
    matrix_dim = ws_filtered.calculate_dimension().split(':')
    rows_count = coordinate_from_string(matrix_dim[1])[1] - 1
    column_count = column_index_from_string(coordinate_from_string(matrix_dim[1])[0]) - 1

    filtered_matrix = np.empty((rows_count, column_count))

    filtered_noun_rows = {}
    filtered_verb_columns = {}

    get_column_names(rows, filtered_verb_columns)
    print('get_column_names completed 2')
    complete_the_loading(rows, filtered_noun_rows, filtered_matrix)
    print('complete_the_loading completed 2')

    rows = ws_soc_pmi.rows
    matrix_dim = ws_soc_pmi.calculate_dimension().split(':')
    rows_count = coordinate_from_string(matrix_dim[1])[1] - 1
    column_count = column_index_from_string(coordinate_from_string(matrix_dim[1])[0]) - 1

    soc_pmi_matrix = np.empty((rows_count, column_count))

    soc_pmi_noun_rows = {}
    soc_pmi_verb_columns = {}

    get_column_names(rows, soc_pmi_verb_columns)
    print('get_column_names completed 2')
    complete_the_loading(rows, soc_pmi_noun_rows, soc_pmi_matrix)
    print('complete_the_loading completed 2')

    content = {'matrix': matrix, 'noun_rows': noun_rows, 'verb_columns': verb_columns,
               'filtered_matrix': filtered_matrix, 'filtered_noun_rows': filtered_noun_rows,
               'filtered_verb_columns': filtered_verb_columns, 'soc_pmi_matrix': soc_pmi_matrix,
               'soc_pmi_noun_rows': soc_pmi_noun_rows, 'soc_pmi_verb_columns': soc_pmi_verb_columns}

    return content


def save_tagged_words(tagged_text, file_name=path_to_txtFolder+'tagged_text.txt', encoding='utf8'):
    """
    Saves a list of tuples in a file. Each tuple contains two strings, the first for the word and the second for
    it's tag. This function is used to generate a file in the txtFiles folder. Use this file to check the POSTagger
    tagging accuracy
    :param tagged_text: list of tuples explained above
    :param file_name: file name to save on
    :param encoding: type of encoding of the file
    :return: None
    """
    f2 = open(file_name, 'w', encoding=encoding)
    for (word, tag) in tagged_text:
        f2.write(word + ' -> ' + tag + '\n')
    f2.close()


def pause():
    input('Press enter to continue')
