import numpy as np
import heapq
from scipy.spatial.distance import cosine

import nltk
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic

import projectFiles.utils as utils

# from pywsd.lesk import cosine_lesk


lch_maximum_obtained_value = 3.63758

class CoocMatrix:
    """
    Class responsible to handle the co-occurrence matrix operations
    """

    def __init__(self, windows=None, build_matrix=True, content=None, enable_lemmatization=True):

        print('matrix construction class')

        #  Dictionary have a noun for key and a row index for value
        self.noun_rows = {}
        self.wordnet_nouns = {}
        self.noun_rows_size = 0
        self.noun_freq = {}


        #  Dictionary have a verb for key and a column index for value
        self.verb_columns = {}
        self.wordnet_verbs = {}
        self.verb_columns_size = 0

        #  The co-occurrence matrix
        self.matrix = np.array([[0, 0], [0, 0]], dtype=float)

        self.verb_filtered_arrays = {}
        self.nouns_from_verb_arrays = {}

        self.is_pmi_calculated = False
        self.soc_pmi_matrix = None

        self.noun_to_noun_sim_matrices = []

        if build_matrix:
            #  Functions called to calculate the co-occurrence matrix and it's filtered version
            self.create_coocmatrix(windows, enable_lemmatization)
            self.filter_coocmatrix2()
        else:
            self.matrix = content['matrix']
            self.verb_columns = content['verb_columns']
            self.verb_columns_size = self.matrix.shape[1]
            self.noun_rows = content['noun_rows']
            self.noun_rows_size = self.matrix.shape[0]

            # self.filtered_matrix = content['filtered_matrix']
            # self.filtered_verb_columns = content['filtered_verb_columns']
            # self.filtered_verb_columns_size = self.filtered_matrix.shape[1]
            # self.filtered_noun_rows = content['filtered_noun_rows']
            # self.filtered_noun_rows_size = self.filtered_matrix.shape[0]

            # The row nouns are exactly equal to the filtered_matrix and the columns are mirrors from the rows in
            # this matrix
            self.soc_pmi_matrix = content['soc_pmi_matrix']

            self.is_pmi_calculated = True

    def create_coocmatrix(self, windows, enable_lemmatization):
        """
        From the list of windows creates the co-occurrence matrix and stores it ans its parameters in self.matrix

        Parameters
        ----------
        windows: List from with the text windows are obtained

        Returns
        -------
        None
        """

        print('create_coocmatrix started')

        lemmatizer = nltk.stem.WordNetLemmatizer()

        window_verbs = []
        window_nouns = []

        progress = 0
        for window in windows:  # Iterates over the windows
            progress += 1
            # if enable_verb_filter:
            #     check_window = list(window)
            #     window_skip_counter = 0
            #
            #     valid_window = False
            #     for (word, tag) in check_window:
            #         if tag.startswith('V'):
            #             word = lemmatizer.lemmatize(word, 'v')
            #             if word in utils.verbs_to_keep:
            #                 valid_window = True
            #                 break
            #
            #     if not valid_window:
            #         window_skip_counter += 1
            #         print('number of skipped windows = ' + str(window_skip_counter))
            #         continue

            central_verb_index = window[1]
            window = window[0]

            # string_from_window = ''
            # for word in window:
            #     string_from_window = string_from_window + ' ' + word[0]



            word_counter = 0
            for (word, tag) in window:  # Iterates over each tagged word inside the window


                if word_counter == central_verb_index and tag.startswith('V'):  # If it is a verb store it on a dictionary and increase the size of
                    # columns
                    if enable_lemmatization:
                        word = lemmatizer.lemmatize(word, 'v')

                    word = 'to ' + word
                    window_verbs.append(word)
                    if word not in self.verb_columns:
                        self.verb_columns[word] = self.verb_columns_size
                        # self.wordnet_verbs[word] = cosine_lesk(string_from_window, word, pos='v')
                        self.verb_columns_size += 1
                        if self.verb_columns_size > 2:
                            self.matrix = np.lib.pad(self.matrix, ((0, 0), (0, 1)), 'constant', constant_values=0)

                if tag.startswith('NN'):  # If it is a Noun do the same as the verbs, but increase the rows

                    if enable_lemmatization:
                        word = lemmatizer.lemmatize(word)

                    if word not in self.noun_freq:
                        self.noun_freq[word] = 1
                    else:
                        self.noun_freq[word] += 1

                    window_nouns.append(word)
                    if word not in self.noun_rows:
                        self.noun_rows[word] = self.noun_rows_size
                        # self.wordnet_nouns[word] = cosine_lesk(string_from_window, word, pos='n')
                        self.noun_rows_size += 1
                        if self.noun_rows_size > 2:
                            self.matrix = np.lib.pad(self.matrix, ((0, 1), (0, 0)), 'constant', constant_values=0)

                word_counter += 1

            # print(string_from_window)
            # for noun in window_nouns:
            #     if self.wordnet_nouns[noun] is not None:
            #         print(noun + ' -> ' + str(self.wordnet_nouns[noun]) + ': ' + self.wordnet_nouns[noun].definition())

            for verb in window_verbs:  # fills the matrix with the co-occurrences
                j = self.verb_columns.get(verb)
                for noun in window_nouns:
                    i = self.noun_rows.get(noun)
                    self.matrix[i][j] += 1

            window_verbs.clear()  # Clear temp lists for next iteration
            window_nouns.clear()

        print('create_coocmatrix ended')


    @staticmethod
    def calc_ppmi(cooc_matrix, improver, constant=0):
        """
        Wights the co-occurrence matrix using PPMI method and some selectable improover to this method

        Parameters
        ----------
        cooc_matrix : numpy matrix, the co-occurrence matrix
        improver : int, selects between none, Laplace smoothing and Palpha
        constant : int, can be the constant of Laplace or Palpha methods

        Returns
        -------
        temp_matrix
        """

        print('calc_ppmi started')

        # Laplace smoothing
        if improver is 1:
            temp_matrix = np.add(cooc_matrix, constant)
        else:
            temp_matrix = cooc_matrix

        # Sum all the elements in the matrix
        total = temp_matrix.sum(dtype=float)
        # Creates an array with each element containing the sum of one column
        total_per_column = temp_matrix.sum(axis=0, dtype='float')
        # Creates an array with each element containing the sum of one row
        total_per_line = temp_matrix.sum(axis=1, dtype='float')

        # Get the matrix dimensions
        (maxi, maxj) = temp_matrix.shape

        # Iterates over all the matrix
        for i in range(maxi):
            for j in range(maxj):

                # Calculates the PMI's constants
                pcw = temp_matrix[i][j]/total
                pw = total_per_line[i]/total
                pc = total_per_column[j]/total

                if improver is 1 and temp_matrix[i][j] - constant == 0:
                    temp_matrix[i][j] = 0

                # Checks for division per zero
                elif pw*pc == 0:
                    temp_matrix[i][j] = 0
                else:
                    # Calculates the new wighted value
                    temp_matrix[i][j] = np.maximum(np.log2(pcw/(pw*pc)), 0)

        print('calc_ppmi ended')

        return temp_matrix

    def filter_coocmatrix2(self):

        print('filter_coocmatrix2 started')

        verb_keys = self.verb_columns.keys()

        num_of_rows = self.matrix.shape[0]
        thirty_percent = int(np.ceil(0.3 * num_of_rows))

        inverted_noun_rows_dict = utils.invert_dictionary(self.noun_rows)

        for verb in verb_keys:
            verb_index = self.verb_columns[verb]

            temp_column = self.matrix[:, verb_index]
            thirty_percent_largest_indices = heapq.nlargest(thirty_percent, range(num_of_rows), temp_column.take)

            most_co_occurring_nouns_values = list(temp_column[thirty_percent_largest_indices])

            self.verb_filtered_arrays[verb] = most_co_occurring_nouns_values
            self.nouns_from_verb_arrays[verb] = [inverted_noun_rows_dict[index] for index
                                                 in thirty_percent_largest_indices]

        print('filter_coocmatrix2 ended')

    # def filter_coocmatrix(self):
    #     """
    #     Finds the 30% highest occurring elements from the matrix, deletes the columns and rows that do not contain
    #     at least one of these elements. Rebuild the matrix storing on the variable "filtered matrix"
    #
    #     Returns
    #     -------
    #     None
    #     """
    #
    #     #  Gets the number that corresponds to 30% of the total of elements in the matrix
    #     thirty_percent = int(np.ceil(self.noun_rows_size * self.verb_columns_size * 0.3))
    #
    #     #  Creates an empty numpy array with the size of the entire matrix
    #     temp_matrix_list = np.empty(self.noun_rows_size * self.verb_columns_size)
    #
    #     x = 0
    #
    #     #  Fit the matrix in a 1D array
    #     for i in self.matrix.flat:
    #         temp_matrix_list[x] = i
    #         x += 1
    #
    #     #  Get a list of indices of the 30% most occurring items on the "1D matrix"
    #     thirty_percent_bigger_ind = heapq.nlargest(thirty_percent,
    #                                                range(self.noun_rows_size * self.verb_columns_size),
    #                                                temp_matrix_list.take)
    #
    #     i = 0
    #     real_aij_matrix_pos = []
    #
    #     #  transform the obtained 1D-matrix's indices to the indices of the real matrix (2D indices)
    #     while i < thirty_percent:
    #         real_aij_matrix_pos.append([thirty_percent_bigger_ind[i] // self.verb_columns_size,
    #                                     thirty_percent_bigger_ind[i] % self.verb_columns_size])
    #         i += 1
    #
    #     self.filtered_verb_columns_size = 0
    #     self.filtered_noun_rows_size = 0
    #     noun_row_idxs = {}
    #     verb_column_idxs = {}
    #
    #     #  Separates the indices in row and column and store them as key in a dictionary to avoid repetition
    #     for two_index in real_aij_matrix_pos:
    #         if two_index[0] not in noun_row_idxs:
    #             noun_row_idxs[two_index[0]] = 0
    #             self.filtered_noun_rows_size += 1
    #         if two_index[1] not in verb_column_idxs:
    #             verb_column_idxs[two_index[1]] = 0
    #             self.filtered_verb_columns_size += 1
    #
    #     k = 0
    #     self.filtered_matrix = np.zeros((self.filtered_noun_rows_size, self.filtered_verb_columns_size), dtype=float)
    #
    #     #  Invert the dictionaries, now they have the index as the key and the word name as the value
    #     temp_dict_noun = dict(zip(self.noun_rows.values(), self.noun_rows.keys()))
    #     temp_dict_verb = dict(zip(self.verb_columns.values(), self.verb_columns.keys()))
    #
    #     #  This loop will build the filtered_matrix iterating over every index of the old matrix and checking if
    #     #  this is one that it can discard.
    #     for i in range(self.noun_rows_size):
    #         if i not in noun_row_idxs:
    #             continue
    #         else:
    #             #  This rebuilds the dictionaries that relate every noun with a row of the matrix
    #             self.filtered_noun_rows[temp_dict_noun[i]] = k
    #             l = 0
    #         for j in range(self.verb_columns_size):
    #             if j not in verb_column_idxs:
    #                 continue
    #             else:
    #                 self.filtered_matrix[k][l] = self.matrix[i][j]
    #                 #  This rebuilds the dictionaries that relate every verb with a column of the matrix
    #                 self.filtered_verb_columns[temp_dict_verb[j]] = l
    #                 l += 1
    #         k += 1

    def cosine_row_sim(self, noun1, noun2):
        """
        This function calculates the cosine similarity between two word vectors
        :param noun1: The name of a noun that belongs to the co-occurrence matrix to be compared
        :param noun2: The name of the other noun to be compared
        :return: The result of the cosine difference
        """
        row_noun2 = self.noun_rows[noun2]
        row_noun1 = self.noun_rows[noun1]

        curr_matrix = self.matrix

        row1 = curr_matrix[row_noun1]
        row2 = curr_matrix[row_noun2]

        return 1 - cosine(row1, row2)

    def cosine_column_sim(self, verb1, verb2, verb_prefix=''):

        verb1 = verb_prefix + verb1
        verb2 = verb_prefix + verb2

        column_verb1 = self.verb_columns[verb1]
        column_verb2 = self.verb_columns[verb2]


        column1 = self.matrix[:, column_verb1]
        column2 = self.matrix[:, column_verb2]

        if verb1 == verb2:
            print(len(column1))


        return 1 - cosine(column1, column2)

    def plot_two_word_vectors(self, noun1_name, noun2_name, verb1_name, verb2_name):
        """
        This function is called by the GUI and focus on obtaining the information needed to load the word vectors
        graphically on the screen. The goal is to display two rows of the matrix to see how they differ when compared
        against two verbs that will be the x and y axis of the plot.
        :param noun1_name: A string containing the name of the first noun to be used in the comparision
        :param noun2_name: A string containing the name of the second noun to be used in the comparision
        :param verb1_name: A string containing the name of the first verb to be used in the comparision (x axis)
        :param verb2_name: A string containing the name of the second verb to be used in the comparision (y axis)
        :return: The function returns a boolean value to indicate to the GUI if the wanted values where found in the
        desired matrix.
        """

        curr_matrix = self.matrix
        curr_noun_dict = self.noun_rows
        curr_verb_dict = self.verb_columns

        # Will try to find the values related to the wanted keys (the keys being the nouns and verbs names and the
        # values been the numerical PPMI weighted values)
        try:
            vector_1 = [curr_matrix[curr_noun_dict[noun1_name]][curr_verb_dict[verb1_name]],
                        curr_matrix[curr_noun_dict[noun1_name]][curr_verb_dict[verb2_name]]]

            vector_2 = [curr_matrix[curr_noun_dict[noun2_name]][curr_verb_dict[verb1_name]],
                        curr_matrix[curr_noun_dict[noun2_name]][curr_verb_dict[verb2_name]]]
        except KeyError:
            return False

        # Calls the utils function for the un-specialized graphical work
        utils.plot_vectors(vector_1, vector_2, noun1_name, noun2_name, verb1_name, verb2_name)
        return True

    def create_soc_pmi_matrix(self, matrix):
        """
        This function has the objective of obtain a second order measurement for the input text. It will generate a
        matrix with zeros from the diagonal to below. Both x and y axis will refer to nouns and each element of the
        part above the diagonal will be the second order co-occurrence value calculated for each noun-noun pair.
        The higher the value, the higher will be the similarity between the two nouns
        :param matrix: will be the numpy matrix used to obtain the soc-pmi-matrix
        :return: None
        """

        print('create_soc_pmi_matrix started')

        # The PPMI values must had been calculated
        if self.is_pmi_calculated is False:
            print('The matrices have to have their pmi values computed before calling this method ')
            return

        # max_value = -1

        # Noun per Noun matrix (much bigger size that the noun-verb matrix)
        self.soc_pmi_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))

        # Default values from the soc-pmi algorithm, after they will need some calculation to be effective
        delta = 0.9
        gama = 3

        # A constant value calculated only here to save processing
        beta_constant = np.log2(matrix.shape[1])/delta

        # This loop makes all the unique noun-noun combinations
        i = 0
        while i < self.soc_pmi_matrix.shape[0] - 1:
            # calculate the bigger loop noun's beta value
            beta1 = np.maximum(int(np.floor(calc_beta(matrix, beta_constant, i))), 1)

            # calculate the number of zero values within this noun's row
            counts = zero_counter(matrix[i])
            zero_counts1 = counts

            # Pre calculates this value to save processing
            beta_1_divisor = np.maximum((beta1-zero_counts1), 1)

            # Gets the indices of the highest values in decreasing order
            beta_largest_pmi1 = heapq.nlargest(beta1, range(matrix.shape[1]), matrix[i].take)

            # This attribution eliminates self noun-noun combinations and repetitive noun-noun combinations, since
            # this matrix will be mirrored by its diagonal if it was fully filled
            j = i + 1

            # This loop combines the noun fixed in the above loop with every other noun in the matrix
            while j < self.soc_pmi_matrix.shape[0]:

                # calculate the smaller loop noun's beta value
                beta2 = np.maximum(int(np.floor(calc_beta(matrix, beta_constant, j))), 1)

                # calculate the number of zero values within this noun's row
                counts = zero_counter(matrix[j])
                zero_counts2 = counts

                # Gets the indices of the highest values in decreasing order
                beta_largest_pmi2 = heapq.nlargest(beta2, range(matrix.shape[1]),
                                                   matrix[j].take)

                # Generates a list of the common values between the two decreasing order highest values in the two
                # rows analyzed by this loop iteration. Since the values in these arrays are the column indices on the
                # matrix we can obtain the second matrix coordinate that will be used below
                common_indices = np.intersect1d(beta_largest_pmi1, beta_largest_pmi2)

                # Will calculate the f_beta value of the algorithm
                f_beta1 = f_beta2 = 0
                for index in common_indices:
                    f_beta1 += np.power(matrix[j][index], gama)
                    f_beta2 += np.power(matrix[i][index], gama)

                # Finally will calculate the second order similarity between the two nouns
                # (i's row noun and j's row noun)
                self.soc_pmi_matrix[i][j] = (f_beta1/beta_1_divisor) + (f_beta2/np.maximum((beta2-zero_counts2), 1))
                # if max_value < self.soc_pmi_matrix[i][j]:
                #     max_value = self.soc_pmi_matrix[i][j]

                j += 1

            i += 1
            print(str(i) + '/' + str(self.soc_pmi_matrix.shape[0] - 1))

        print('create_soc_pmi_matrix ended')

    def highest_cosine_sim_array(self):

        print('highest_cosine_sim_array started')

        noun_dict = self.noun_rows

        noun_dict_size = len(noun_dict)
        noun_dict_keys = list(noun_dict.keys())

        highest_sim_nouns = []

        only_value_array_size = ((noun_dict_size * noun_dict_size) - noun_dict_size) // 2
        only_value_array = np.empty(only_value_array_size)

        counter = 0
        i = 0
        while i < noun_dict_size - 1:
            j = i + 1

            while j < noun_dict_size:
                cosine_sim_value = self.cosine_row_sim(noun_dict_keys[i], noun_dict_keys[j])
                highest_sim_nouns.append((noun_dict_keys[i], noun_dict_keys[j], cosine_sim_value))
                only_value_array[counter] = cosine_sim_value
                counter += 1

                j += 1

            print(str(i) + '/' + str(noun_dict_size))
            i += 1

        largest_sim_indices = heapq.nlargest(only_value_array_size, range(only_value_array_size), only_value_array.take)
        print(1)

        print('highest_cosine_sim_array ended')

        return [highest_sim_nouns[i] for i in largest_sim_indices]

    def calculate_sim_matrix(self):

        print('calculate_sim_matrix started')

        self.noun_to_noun_sim_matrices.append(np.add(np.zeros((self.noun_rows_size, self.noun_rows_size), dtype=float),
                                                     0.01))
        self.noun_to_noun_sim_matrices.append(np.add(np.zeros((self.noun_rows_size, self.noun_rows_size), dtype=float),
                                                     0.01))
        self.noun_to_noun_sim_matrices.append(np.add(np.zeros((self.noun_rows_size, self.noun_rows_size), dtype=float),
                                                     0.01))
        self.noun_to_noun_sim_matrices.append(np.add(np.zeros((self.noun_rows_size, self.noun_rows_size), dtype=float),
                                                     0.01))
        self.noun_to_noun_sim_matrices.append(np.add(np.zeros((self.noun_rows_size, self.noun_rows_size), dtype=float),
                                                     0.01))

        inverted_noun_dict = utils.invert_dictionary(self.noun_rows)

        brown_ic = wordnet_ic.ic('ic-brown.dat')

        for key in inverted_noun_dict:
            print(str(key) + ': ' + inverted_noun_dict[key])

        i = 0
        while i < (self.noun_rows_size - 1):
            j = i + 1
            w1 = wordnet.synsets(inverted_noun_dict[i], pos=wordnet.NOUN)
            if not w1:
                print('Not able to find this noun: ' + inverted_noun_dict[i])
                i += 1
                continue

            w1 = w1[0]

            while j < self.noun_rows_size:
                w2 = wordnet.synsets(inverted_noun_dict[j], pos=wordnet.NOUN)
                if not w2:
                    j += 1
                    continue

                w2 = w2[0]

                value = w1.wup_similarity(w2)
                value = utils.limit_value(value, 0.01, 1.0)
                self.noun_to_noun_sim_matrices[0][i][j] = value

                value = w1.lch_similarity(w2)/lch_maximum_obtained_value
                value = utils.limit_value(value, 0.01, 1.0)
                self.noun_to_noun_sim_matrices[1][i][j] = value

                value = w1.jcn_similarity(w2, brown_ic)
                value = utils.limit_value(value, 0.01, 1.0, True)
                self.noun_to_noun_sim_matrices[2][i][j] = value


                value = w1.lin_similarity(w2, brown_ic)
                value = utils.limit_value(value, 0.01, 1.0)
                self.noun_to_noun_sim_matrices[3][i][j] = value

                value = (self.noun_to_noun_sim_matrices[0][i][j] +
                                       self.noun_to_noun_sim_matrices[1][i][j] +
                                       self.noun_to_noun_sim_matrices[2][i][j] +
                                       self.noun_to_noun_sim_matrices[3][i][j]) / 4.0

                value = utils.limit_value(value, 0.01, 1.0)

                self.noun_to_noun_sim_matrices[4][i][j] = value

                j += 1

            print('sim_matrix: ' + str(i) + '\n')
            i += 1

        print('calculate_sim_matrix ended')

    def get_20_percent_of_highest_pairs(self):
        # Gets the number that corresponds to 20% of the total of elements in the matrix
        twenty_percent = int(np.ceil(self.noun_rows_size * self.verb_columns_size * 0.2))

        #  Creates an empty numpy array with the size of the entire matrix
        temp_matrix_list = np.empty(self.noun_rows_size * self.verb_columns_size)

        x = 0

        #  Fit the matrix in a 1D array
        for i in self.matrix.flat:
            temp_matrix_list[x] = i
            x += 1

        #  Get a list of indices of the 20% most occurring items on the "1D matrix"
        thirty_percent_bigger_ind = heapq.nlargest(twenty_percent,
                                                   range(self.noun_rows_size * self.verb_columns_size),
                                                   temp_matrix_list.take)

        i = 0
        real_aij_matrix_pos = []

        #  transform the obtained 1D-matrix's indices to the indices of the real matrix (2D indices)
        while i < twenty_percent:
            real_aij_matrix_pos.append([thirty_percent_bigger_ind[i] // self.verb_columns_size,
                                        thirty_percent_bigger_ind[i] % self.verb_columns_size])
            i += 1

        inverted_noun_rows = utils.invert_dictionary(self.noun_rows)
        inverted_verb_columns = utils.invert_dictionary(self.verb_columns)

        highest_values_list = []
        for i,j in real_aij_matrix_pos:
            highest_values_list.append((self.matrix[i][j], inverted_noun_rows[i], inverted_verb_columns[j]))

        return highest_values_list


def calc_beta(matrix, beta_constant, row_index):
    """
    Calculates the beta value of the algorithm in a different way of the one shown in the paper. This is easier and
    will only impact in a different number of elements in the 'beta_largest_pmi*' vector size. The constants can be
    calibrated to compensate for a difference in the beta value.
    :param matrix: Matrix from which the value of the row will be extracted
    :param beta_constant: The beta constant value to avoid calculate it every time
    :param row_index: Which row to pick
    :return: The beta value
    """
    # This formula was taken from the original paper. Although, the sum here uses the already weighted pmi values and
    # in the paper the sum will use the raw term frequency. I think there is no problem in that.
    return np.power(np.log(matrix[row_index].sum()), 2) * beta_constant


def zero_counter(vec):
    """
    Will count the number of zero values inside an array
    :param vec: Array from which the number of zero values will be calculated
    :return: The number of zero values inside the array 'vec'
    """
    counter = 0
    for element in vec:
        if element == 0:
            counter += 1

    return counter


def save_noun_similarity_array(file_name, path_dict, encoding, noun_sim_array, isChapter=False):
    if isChapter:
        path = path_dict['path_to_output_txt_chapters']
    else:
        path = path_dict['path_to_output_txt']

    output_file = open(path + file_name, 'w', encoding=encoding)

    for (noun1, noun2, sim_value) in noun_sim_array:
        output_file.write(noun1 + ' with ' + noun2 + ': ' + str(sim_value) + '\n')

    output_file.close()


def calculate_sim_matrix_from_dif_lists(word_list_row, word_list_column):
    i_size = len(word_list_row) - 1
    j_size = len(word_list_column)

    unknown_words = {}

    avg_matrix = np.add(np.zeros((len(word_list_row), len(word_list_column)), dtype=float),
                                                             0.001)

    brown_ic = wordnet_ic.ic('ic-brown.dat')

    i = 0
    while i < i_size:
        w1 = wordnet.synset(word_list_row[i])
        j = i + 1
        while j < j_size:
            w2 = wordnet.synsets(word_list_column[j], 'v')
            if not w2:
                j += 1
                unknown_words[word_list_column[j]] = 1
                continue
            w2 = w2[0]

            value = 0
            value += utils.limit_value(w1.wup_similarity(w2), 0.001, 1.0)
            value += utils.limit_value(w1.jcn_similarity(w2, brown_ic), 0.001, 1.0, True)
            value += utils.limit_value(w1.lin_similarity(w2, brown_ic), 0.001, 1.0)
            value += utils.limit_value(w1.lch_similarity(w2) / 3.258096538021482, 0.001, 1.0)
            value /= 4

            avg_matrix[i][j] = value
            j += 1
        i+= 1

    return unknown_words, avg_matrix

def calculate_sim_matrix_from_list(word_list, methods_list, word_pos='n', full_synsets=False, all_matrix=False):

    print('calculate_sim_matrix_from_list started')

    content_dict = {}
    noun_to_noun_sim_matrices = {}
    unknown_words = {}

    word_list_size = len(word_list)
    for method in methods_list:
        noun_to_noun_sim_matrices[method] = np.add(np.zeros((word_list_size, word_list_size), dtype=float),
                                                            0.001)

    noun_to_noun_sim_matrices['average_of_methods'] = np.add(np.zeros((word_list_size, word_list_size), dtype=float),
                                               0.001)

    brown_ic = wordnet_ic.ic('ic-brown.dat')

    i = 0
    if all_matrix:
        bigger_loop_limit = word_list_size
    else:
        bigger_loop_limit = (word_list_size - 1)

    while i < bigger_loop_limit:

        if all_matrix:
            j = 0
        else:
            j = i + 1

        if full_synsets:
            w1 = wordnet.synset(word_list[i])
        else:
            w1 = wordnet.synsets(word_list[i], word_pos)

            if not w1:
                print('Not able to find this noun: ' + word_list[i])
                unknown_words[word_list[i]] = False
                i += 1
                continue

            w1 = w1[0]

        while j < word_list_size:

            if full_synsets:
                w2 = wordnet.synset(word_list[j])
            else:
                w2 = wordnet.synsets(word_list[j], word_pos)

                if not w2:
                    j += 1
                    continue

                w2 = w2[0]

            if 'wup' in noun_to_noun_sim_matrices:
                value = w1.wup_similarity(w2)
                value = utils.limit_value(value, 0.001, 1.0)
                noun_to_noun_sim_matrices['wup'][i][j] = value

            if 'jcn' in noun_to_noun_sim_matrices:
                value = w1.jcn_similarity(w2, brown_ic)
                value = utils.limit_value(value, 0.001, 1.0, True)
                noun_to_noun_sim_matrices['jcn'][i][j] = value

            if 'lin' in noun_to_noun_sim_matrices:
                value = w1.lin_similarity(w2, brown_ic)
                value = utils.limit_value(value, 0.001, 1.0)
                noun_to_noun_sim_matrices['lin'][i][j] = value

            if 'lch' in noun_to_noun_sim_matrices:
                value = w1.lch_similarity(w2)
                if word_pos == 'n':
                    value = value / 3.6375861597263857
                else:
                    value = value / 3.258096538021482
                value = utils.limit_value(value, 0.001, 1.0)
                noun_to_noun_sim_matrices['lch'][i][j] = value


            value = 0.0
            for method in methods_list:
                value += noun_to_noun_sim_matrices[method][i][j]

            value = value/len(methods_list)

            value = utils.limit_value(value, 0.001, 1.0)
            noun_to_noun_sim_matrices['average_of_methods'][i][j] = value

            j += 1

        i += 1
        print('calculate_sim_matrix_from_list: ' + str(i) + '/' + str(word_list_size-1))

    print('calculate_sim_matrix_from_list ended')

    content_dict['noun_to_noun_sim_matrices'] = noun_to_noun_sim_matrices
    content_dict['unknown_words'] = unknown_words

    return content_dict
