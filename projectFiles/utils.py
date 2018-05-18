"""
This file intends to be an utility box, containing functions to help with smaller functionalities
"""

import projectFiles.constants as cts

import numpy as np
import heapq

import os, signal
from subprocess import Popen

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import textwrap as tw

import nltk
from nltk.tag.stanford import CoreNLPPOSTagger

from py4j.java_gateway import JavaGateway


class StanfordProcess:
    def __init__(self, path_to_script):
        self.path_to_script = path_to_script
        self.pid = 0

    def start_process(self):
        self.pid = Popen(self.path_to_script, shell=True)

    def kill_process(self):
        os.kill(int(self.pid), signal.SIGTERM)


'''
Below are functions called by the main.py file and are the start of the noun extraction process from books and xlsx
sheets
'''


def read_text_input(file_input_path, encoding, lowerText=False):
    print("read_text_input started")

    # This will open the text .txt archive and read it
    file = open(file_input_path, 'r', encoding=encoding)

    # This will transform the archive into a huge string
    raw_text = file.read()

    if lowerText:
        # This will lower all the upper case letters in the string (in the entire input file text)
        raw_text = raw_text.lower()

    print("read_text_input ended")

    return raw_text


def tokenize_string(text_string, eliminateNonAlphaNumericalCharacters=False):
    print("tokenize_string started")

    if eliminateNonAlphaNumericalCharacters:
        # This will create a python list named 'tokens' that will have each word/number as an element
        # It will eliminate all kinds of non-alpha numerical characters (punctuation included)
        tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text_string)
    else:
        tokens = nltk.tokenize.word_tokenize(text_string)

    print("tokenize_string ended")

    return tokens


def tag_tokens_using_stanford_corenlp(token_list, corenlp_server_address='http://localhost:9000'):
    # print("tag_tokens_using_stanford_corenlp started")

    tagger = CoreNLPPOSTagger(url=corenlp_server_address)

    # The piece of code below is exists to deal with a limitation of the Stanford's coreNLP Server that only
    # supports 100000 characters per server call. So this will break the text in a lot of smaller pieces and send
    # them to the server and after will unite them all in one list of tagged words ('tagged_text')
    tagged_text = []
    txt_size = len(token_list)
    i = 0
    while i < txt_size:

        if i + 6000 >= txt_size:
            tokens_to_tag = token_list[i:txt_size]
            i = txt_size
        else:
            tokens_to_tag = token_list[i:i + 6000]
            i += 6000

        tagged_text += tagger.tag(tokens_to_tag)

    # print("tag_tokens_using_stanford_corenlp ended")

    return tagged_text


def tokens_to_windows(tokens, window_size):
    """
    Transform a tokenized text into a series of windows for them to serve as contexts
    :param tokens: A python list containing the tokenized text
    :param window_size: The wanted window size
    :return: A python list containing the "tokens" separated in windows
    """

    print("tokens_to_windows started")

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

    print("tokens_to_windows ended")

    return windows


def tokens_to_centralized_windows(tagged_text, window_size, enable_verb_filter):
    """
    This function is other windowing method. It will find the desired verbs and make windows around them. The windows
    can overlap, its normal.
    :param tagged_text: A list of tuples. The contents of the tuples are two string, the first being the word and the
    second being the tag related to the word
    :param window_size: It is an integer that indicates the wanted size of the windows
    :return: Returns a list of lists. Each inner list is a window containing the word-tag tuples
    """

    print("tokens_to_centralized_windows started")

    content_dict = {}

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

    verb_frequency_dict = {}

    i = 0
    # Iterates over the tagged_text and create the windows based on it
    while i < tagged_text_size:

        lemmatized_word = lemmatizer.lemmatize(tagged_text[i][0], 'v')

        # Checks if the current word is a verb and checks if it's infinitive form is present on the dict of wanted verbs
        if tagged_text[i][1].startswith('V') and (not enable_verb_filter or lemmatized_word in cts.verbs_to_keep):

            if (lemmatized_word in verb_frequency_dict) == False:
                verb_frequency_dict[lemmatized_word] = 1
            else:
                verb_frequency_dict[lemmatized_word] += 1

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
            temp_list = [tagged_text[start:end], i-start]
            windows.append(temp_list.copy())

        i += 1

    content_dict['windows'] = windows
    content_dict['verb_frequency_dict'] = verb_frequency_dict

    print("tokens_to_centralized_windows ended")

    return content_dict


def save_verb_frequency(verb_frequency_dict, path_to_output_txt, feature_name,encoding):
    list_of_tuples = sort_dict(verb_frequency_dict)
    save_list_of_tuples(list_of_tuples, path_to_output_txt, 'verb_frequency' + feature_name, encoding)


def save_highest_elements_on_coocmatrix(cooc_matrix, path_to_output_txt, feature_name, encoding):
    save_list_of_tuples(cooc_matrix.get_20_percent_of_highest_pairs(), path_to_output_txt,
                              '20PerCentHighestPairs' + feature_name, encoding)


def save_noun_frequency(noun_frequency_dict, path_to_output_txt, feature_name, encoding):
    list_of_tuples = sort_dict(noun_frequency_dict)
    save_list_of_tuples(list_of_tuples, path_to_output_txt, 'nouns_frequency' + feature_name, encoding)


def calculate_generate_sim_matrix_gdf(cooc_matrix, path_dict, calculate_sim_matrix_method, min_edge_value):
    inverted_noun_rows = invert_dictionary(cooc_matrix.noun_rows)
    temp_noun_rows_list = [inverted_noun_rows[i] for i in range(len(inverted_noun_rows))]
    content_dict = calculate_sim_matrix_method(temp_noun_rows_list, cts.all_semantic_similarity_methods)

    cooc_matrix.noun_to_noun_sim_matrices = content_dict['noun_to_noun_sim_matrices']
    unknown_words = content_dict['unknown_words']

    save_noun_sim_matrix_in_gdf(cooc_matrix, unknown_words, cooc_matrix.noun_rows,
                                      cts.all_semantic_similarity_methods + ['average_of_methods'], 'simGraph',
                                      path_dict, True, min_edge_value)


'''
Below are functions that deal with the reading, opening and saving of xlsx archives
'''



'''
Below are functions related to graphics
'''


def plot_vectors(vec1_coord, vec2_coord, vec1_name, vec2_name, verb1_name, verb2_name):
    # Pyplot method to clear the old drawings
    plt.gcf().clear()

    # Calculate the module of the arrays
    vec1_module = np.sqrt(np.power(vec1_coord[0], 2) + (np.power(vec1_coord[1], 2)))
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


def create_bar_plot(file_path, file_name, object_list, y_value_list, title, y_label, text_below='', figsize=(10, 3),
                    alpha=0.7, align='center', width=0.5):

    y_pos = np.arange(len(object_list))

    plt.figure(figsize=figsize)

    plt.bar(y_pos, y_value_list, align=align, alpha=alpha, width=width)
    # line_plot = plt.twinx()
    plt.plot(y_pos, y_value_list, 'bo', y_pos, y_value_list, 'k')
    plt.grid(b=False)

    fig_txt = tw.fill(tw.dedent(text_below.rstrip()))
    plt.figtext(0.5, -0.2, fig_txt, horizontalalignment='center', fontsize=10, multialignment='left',
                bbox=dict(boxstyle="round", facecolor='#D8D8D8', ec="0.5", pad=0.5, alpha=1), fontweight='bold')

    plt.xticks(y_pos, object_list)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(file_path + file_name, bbox_inches='tight')
    plt.close()


def create_cognitive_level_distribution_graph(path_to_img_folder, name_of_img, verb_frequency_dict):

    cognitive_level_frequency = {}
    for (verb,frequency) in verb_frequency_dict.items():
        verb_cognitive_level = get_verb_cognitive_level(verb)
        if verb_cognitive_level is None:
            continue

        if verb_cognitive_level not in cognitive_level_frequency:
            cognitive_level_frequency[verb_cognitive_level] = frequency
        else:
            cognitive_level_frequency[verb_cognitive_level] += frequency

    object_list = cts.names_of_cognitive_levels
    y_values = sort_cognitive_levels_associated_values(cognitive_level_frequency)

    create_bar_plot(path_to_img_folder, name_of_img, object_list, y_values, 'Verbs in each cognitive level',
                    'frequency of verbs')

    txt_out = open(path_to_img_folder + 'text_distribution', 'w')
    for i in range(len(y_values)):
        txt_out.write(object_list[i] + ' - ' + str(y_values[i]) + '\n')
    txt_out.close()


'''
Below are functions related to saving in text archives
'''


def save_tagged_words(tagged_text, path_dict,
                      encoding='utf8'):
    """
    Saves a list of tuples in a file. Each tuple contains two strings, the first for the word and the second for
    it's tag. This function is used to generate a file in the txtFiles folder. Use this file to check the POSTagger
    tagging accuracy
    :param tagged_text: list of tuples explained above
    :param file_name: file name to save on
    :param encoding: type of encoding of the file
    :return: None
    """
    f2 = open(path_dict['path_to_output_txt'], 'w', encoding=encoding)
    for (word, tag) in tagged_text:
        f2.write(word + ' -> ' + tag + '\n')
    f2.close()


def save_list_of_tuples(list_of_tuples, folder_path, source_name, encoding='utf8'):

    output_file = open(folder_path + source_name + '.txt', 'w',
                       encoding=encoding)
    for tuple in list_of_tuples:
        temp_string = ''
        for value in tuple:
            temp_string += '-' + str(value)
        output_file.write(temp_string[1:] +  '\n')


'''
Below are functions related to create gdf archives 
'''


def save_noun_sim_matrix_in_gdf(cooc_matrix, unknown_words, noun_dict, methods, book_name, path_dict,
                                eliminate_unknown_words=True, edge_min_value=0.0):

    edge_min_value = limit_value(edge_min_value, 0.0, 1.0)

    output_files = []
    for method in methods:
        output_files.append(open(path_dict['path_to_output_gdf'] + book_name + '_' + method + '.gdf', 'w'))

    for output_file_index in range(len(output_files)):
        # print('Creating the graph archives ... current progress = ' + str(output_file_index) + '/' +
        # str(len(output_files)))

        output_files[output_file_index].write('nodedef>name VARCHAR')

        for i in range(17):
            output_files[output_file_index].write(', verb' + str(i) + ' VARCHAR')

        output_files[output_file_index].write('\n')

        curr_matrix = cooc_matrix.noun_to_noun_sim_matrices[methods[output_file_index]]
        inverted_noun_dict = invert_dictionary(noun_dict)

        i = 0
        check_again = []
        while i < curr_matrix.shape[0] - 1:
            j = i + 1
            has_above_min = False
            while j < curr_matrix.shape[0]:
                if curr_matrix[i][j] > edge_min_value:
                    check_again.append(inverted_noun_dict[j])
                    has_above_min = True
                    break
                j += 1

            if has_above_min == False:
                unknown_words[inverted_noun_dict[i]] = False

            i += 1

        for noun in check_again:
            if noun in unknown_words:
                print(noun)
                del unknown_words[noun]

        for noun_key in noun_dict:
            if eliminate_unknown_words and noun_key in unknown_words:
                continue

            output_files[output_file_index].write(noun_key)
            still_have_verbs_to_fill = 16
            for verb_key, verb_column in cooc_matrix.verb_columns.items():
                curr_row = cooc_matrix.noun_rows[noun_key]
                if cooc_matrix.matrix[curr_row][verb_column] > 0 and still_have_verbs_to_fill > 0:
                    output_files[output_file_index].write(', ' + verb_key)
                    still_have_verbs_to_fill -= 1

            for i in range(still_have_verbs_to_fill):
                output_files[output_file_index].write(', ')

            output_files[output_file_index].write('\n')

        output_files[output_file_index].write('edgedef>node1 VARCHAR, node2 VARCHAR, weight FLOAT\n')

        i = 0
        while i < curr_matrix.shape[0] - 1:

            if eliminate_unknown_words and inverted_noun_dict[i] in unknown_words:
                i += 1
                continue

            j = i + 1
            while j < curr_matrix.shape[0]:

                if eliminate_unknown_words and inverted_noun_dict[j] in unknown_words:
                    j += 1
                    continue

                if edge_min_value > curr_matrix[i][j]:
                    j += 1
                    continue

                output_files[output_file_index].write(inverted_noun_dict[i] + ',' + inverted_noun_dict[j] + ',' +
                                                      '{0:.2f}'.format(curr_matrix[i][j]) + '\n')
                j += 1

            i += 1


def save_noun_sim_matrix_in_gdf_2(noun_to_noun_sim_matrices, noun_list, department_list, full_noun_and_verb_list,
                                  synset_list, methods, path, name, eliminate_same_department_edges=True):
    # if eliminate_same_department_edges:
    #     dept_edges = "0"
    # else:
    #     dept_edges = '1'

    output_files = []
    for method in methods:
        output_files.append(open(path + name + '_' + method + '.gdf', 'w'))
        # output_files.append(open(path + name + '_' + method + '_' + dept_edges + '.gdf', 'w'))

    for output_file_index in range(len(output_files)):
        # print('Creating the graph archives ... current progress = ' + str(output_file_index+1) + '/' +
        #       str(len(output_files)))

        output_files[output_file_index].write('nodedef>name VARCHAR, reducedName VARCHAR, synset VARCHAR,'
                                              ' verb VARCHAR, color VARCHAR\n')

        for i in range(len(noun_list)):
            output_files[output_file_index].write(full_noun_and_verb_list[i][0] + ',' + noun_list[i] + ',' +
                                                  str(synset_list[i]) + ',' + full_noun_and_verb_list[i][1])
            department = department_list[i]

            if department in cts.department_colors:
                color = cts.department_colors[department]
            else:
                color = "#000000"

            output_files[output_file_index].write("," + color + "\n")

        output_files[output_file_index].write('edgedef>node1 VARCHAR, node2 VARCHAR, weight FLOAT\n')

        curr_matrix = noun_to_noun_sim_matrices[methods[output_file_index]]

        print("matrixShape = " + str(curr_matrix.shape[0]))

        i = 0
        while i < curr_matrix.shape[0] - 1:

            j = i + 1
            while j < curr_matrix.shape[0]:

                if eliminate_same_department_edges and department_list[i] == department_list[j]:
                    j += 1
                    continue

                output_files[output_file_index].write(full_noun_and_verb_list[i][0] + ',' +
                                                      full_noun_and_verb_list[j][0] + ',' +
                                                      '{0:.2f}'.format(curr_matrix[i][j]) + '\n')
                j += 1

            i += 1


def save_noun_sim_matrix_in_gdf3(sim_matrices, unknown_words, noun_dict, noun_list, methods, book_name, path_dict,
                                 eliminate_unknown_words=True, edge_min_value=0.0):

    edge_min_value = limit_value(edge_min_value, 0.0, 1.0)

    output_files = []
    for method in methods:
        output_files.append(open('/home/paulojeunon/Documents/' + book_name + '_' + method + '.gdf', 'w'))

    for output_file_index in range(len(output_files)):

        output_files[output_file_index].write('nodedef>name VARCHAR, color VARCHAR')

        for i in range(17):
            output_files[output_file_index].write(', verb' + str(i) + ' VARCHAR')

        output_files[output_file_index].write('\n')

        curr_matrix = sim_matrices[methods[output_file_index]]

        for i in range(len(noun_list)):
            has_above_min = False
            for j in range(len(noun_list)):
                if curr_matrix[i][j] > edge_min_value:
                    has_above_min = True
                    break

            if has_above_min == False:
                unknown_words[noun_list[i]] = False

        for noun_key in noun_list:
            if eliminate_unknown_words and noun_key in unknown_words:
                continue

            output_files[output_file_index].write(noun_key)
            output_files[output_file_index].write(', ' + noun_dict[noun_key]['my_color'])

            for verb in noun_dict[noun_key].keys():
                if verb.startswith('my_c'):
                    continue
                output_files[output_file_index].write(', ' + verb)

            for i in range(17 - len(noun_dict[noun_key])):
                output_files[output_file_index].write(', ')

            output_files[output_file_index].write('\n')

        output_files[output_file_index].write('edgedef>node1 VARCHAR, node2 VARCHAR, weight FLOAT\n')

        i = 0
        while i < curr_matrix.shape[0] - 1:

            if eliminate_unknown_words and noun_list[i] in unknown_words:
                i += 1
                continue

            j = i + 1
            while j < curr_matrix.shape[0]:

                if eliminate_unknown_words and noun_list[j] in unknown_words:
                    j += 1
                    continue

                if edge_min_value > curr_matrix[i][j]:
                    j += 1
                    continue

                output_files[output_file_index].write(noun_list[i] + ',' + noun_list[j] + ',' +
                                                      '{0:.2f}'.format(curr_matrix[i][j]) + '\n')
                j += 1

            i += 1


'''
Below are general use functions
'''


def invert_dictionary(dictionary):
    return dict(zip(dictionary.values(), dictionary.keys()))


def limit_value(value, min_value, max_value, is_jcn=False):
    if value > max_value:
        value = max_value
    elif value < min_value:
        if is_jcn:
            if value == 1e-300:
                value = 1
            else:
                value = min_value
        else:
            value = min_value

    return value


def create_new_directory(path_plus_dir_name):
    if not os.path.exists(path_plus_dir_name):
        os.makedirs(path_plus_dir_name, exist_ok=True)


def sort_dict(dict):
    dict_size = len(dict)
    key_array = []
    value_array = np.empty(dict_size, dtype=int)

    i = 0
    for key,value in dict.items():

        key_array.append(key)

        if isinstance(value, list):
            value_array[i] = np.sum(value)
        elif isinstance(value, int):
            value_array[i] = value

        i += 1

    list_of_tuples = sort_and_match(value_array, key_array)

    return list_of_tuples


def sort_and_match(list_of_values, list_to_be_matched):
    sorted_value_indices = heapq.nlargest(list_of_values.shape[0], range(list_of_values.shape[0]), list_of_values.take)

    list_of_tuples = []
    for index in sorted_value_indices:
        list_of_tuples.append((list_to_be_matched[index], list_of_values[index]))

    return list_of_tuples


def get_verb_cognitive_level(verb, with_verbs_at_end=True):
    for level, verbs in cts.cognitive_levels.items():
        if verb in verbs:
            if with_verbs_at_end:
                return level
            else:
                return level[:-6]


def get_book_names_in_json(get_all_books_combined):
    book_name_list = []

    for name in cts.data.keys():
        if name != 'interview' and (get_all_books_combined or name != 'all_books_combined'):
            book_name_list.append(name)

    return book_name_list


def sort_cognitive_levels_associated_values(cognitive_levels_dict, extra_name='_verbs'):
    return [cognitive_levels_dict['knowledge' + extra_name], cognitive_levels_dict['comprehension' + extra_name],
            cognitive_levels_dict['application' + extra_name], cognitive_levels_dict['analysis' + extra_name],
            cognitive_levels_dict['synthesis' + extra_name], cognitive_levels_dict['evaluation' + extra_name]]


'''
Below are Java interface functions
'''


def execute_java(stage_dict, all_stages):
    gateway = JavaGateway()
    word_graph_handler = gateway.entry_point
    stage_hashmap = gateway.jvm.java.util.HashMap()
    pathDict = gateway.jvm.java.util.HashMap()

    for (key,value) in cts.data['interview'].items():
        pathDict.put(key, value)

    wordCoder = word_graph_handler.wordCoder

    for (stage_name, content_dict) in stage_dict.items():
        word_container_hashmap = gateway.jvm.java.util.HashMap()

        for i in range(len(content_dict['noun_list'])):
            word_container = gateway.jvm.WordContainer()
            word_container.setFullWord(content_dict['full_noun_and_verb_list'][i][0])
            word_container.setVerb(content_dict['full_noun_and_verb_list'][i][1])
            word_container.setReducedWord(content_dict['noun_list'][i])
            word_container.setSynset(str(content_dict['synset_list'][i]).strip())
            word_container.setNatureOfEntity(content_dict['nature_of_entities_list'][i])

            curr_dept = content_dict['department_list'][i]
            if curr_dept in cts.department_colors:
                color = cts.department_colors[curr_dept]
            else:
                color = "#000000"

            word_container.setHexColor(color)

            word_container_hashmap.put(wordCoder(word_container.getReducedWord(), word_container.getSynset()),
                                       word_container)

        stage_hashmap.put(stage_name, word_container_hashmap)

    word_graph = word_graph_handler.getWordGraph(stage_hashmap,
                                                 pathDict)

    word_graph.wordGraphCreationHandler(all_stages)


'''
Below here are some pre-steps that have to be made
'''

def setup_environment():
    for dict in cts.data.values():
        for key,value in dict.items():
            if isinstance(key, str) and isinstance(value, str) and key != 'department':
                create_new_directory(value)
