import numpy as np
import nltk
import heapq
import utils
from scipy.spatial.distance import cosine


class CoocMatrix:
    """
    Class responsible to handle the co-occurrence matrix operations
    """

    def __init__(self, windows=None, build_matrix=True, content=None):
        #  Dictionary have a noun for key and a row index for value
        self.noun_rows = {}
        self.noun_rows_size = 0

        #  Dictionary have a verb for key and a column index for value
        self.verb_columns = {}
        self.verb_columns_size = 0

        #  The co-occurrence matrix
        self.matrix = np.array([[0, 0], [0, 0]])

        #  The same thing as the above variables, but, now, referring to the filtered matrix
        self.filtered_matrix = 0
        self.filtered_verb_columns_size = 0
        self.filtered_noun_rows_size = 0
        self.filtered_verb_columns = {}
        self.filtered_noun_rows = {}

        self.is_pmi_calculated = False
        self.soc_pmi_matrix = None

        if build_matrix:
            #  Functions called to calculate the co-occurrence matrix and it's filtered version
            self.create_coocmatrix(windows)
            self.filter_coocmatrix()
        else:
            self.matrix = content['matrix']
            self.verb_columns = content['verb_columns']
            self.verb_columns_size = self.matrix.shape[1]
            self.noun_rows = content['noun_rows']
            self.noun_rows_size = self.matrix.shape[0]

            self.filtered_matrix = content['filtered_matrix']
            self.filtered_verb_columns = content['filtered_verb_columns']
            self.filtered_verb_columns_size = self.filtered_matrix.shape[1]
            self.filtered_noun_rows = content['filtered_noun_rows']
            self.filtered_noun_rows_size = self.filtered_matrix.shape[0]


    def create_coocmatrix(self, windows):
        """
        From the list of windows creates the co-occurrence matrix and stores it ans its parameters in self.matrix

        Parameters
        ----------
        windows: List from with the text windows are obtained

        Returns
        -------
        None
        """

        lemmatizer = nltk.stem.WordNetLemmatizer()

        window_verbs = []
        window_nouns = []

        for window in windows:  # Iterates over the windows
            check_window = list(window)

            valid_window = False
            for (word, tag) in check_window:
                if tag.startswith('V'):
                    word = lemmatizer.lemmatize(word, 'v')
                    if word in utils.verbs_to_keep:
                        valid_window = True
                        break

            if not valid_window:
                continue

            for (word, tag) in window:  # Iterates over each tagged word inside the window
                if tag.startswith('V'):  # If it is a verb store it on a dictionary and increase the size of
                    # columns
                    word = lemmatizer.lemmatize(word, 'v')

                    if word in utils.verbs_to_keep:
                        word = 'to ' + word
                        window_verbs.append(word)
                        if word not in self.verb_columns:
                            self.verb_columns[word] = self.verb_columns_size
                            self.verb_columns_size += 1
                            if self.verb_columns_size > 2:
                                self.matrix = np.lib.pad(self.matrix, ((0, 0), (0, 1)), 'constant', constant_values=0)

                if tag.startswith('NN'):  # If it is a Noun do the same as the verbs, but increase the rows
                    window_nouns.append(word)
                    if word not in self.noun_rows:
                        self.noun_rows[word] = self.noun_rows_size
                        self.noun_rows_size += 1
                        if self.noun_rows_size > 2:
                            self.matrix = np.lib.pad(self.matrix, ((0, 1), (0, 0)), 'constant', constant_values=0)

            for verb in window_verbs:  # fills the matrix with the co-occurrences
                j = self.verb_columns.get(verb)
                for noun in window_nouns:
                    i = self.noun_rows.get(noun)
                    self.matrix[i][j] += 1

            window_verbs.clear()  # Clear temp lists for next iteration
            window_nouns.clear()

        # temp_inv_noun_dict =  utils.invert_dictionary(self.noun_rows)

        # i = 0
        # while i < self.matrix.shape[0]:
        #     if np.sum(self.matrix[i]) == 0:
        #         self.matrix = np.delete(self.matrix,i,0)
        #         del temp_inv_noun_dict[i]
        #         i -= 1
        #
        #     i += 1
        #
        # self.noun_rows_size = self.matrix.shape[0]



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
        # Laplace smoothing
        if improver is 1:
            temp_matrix = np.add(cooc_matrix, constant)
        else:
            temp_matrix = cooc_matrix

        # Sum all the elements in the matrix
        total = temp_matrix.sum()
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

                # Checks for division per zero
                if pw*pc == 0:
                    temp_matrix[i][j] = 0
                else:
                    # Calculates the new wighted value
                    temp_matrix[i][j] = np.maximum(np.log2(pcw/(pw*pc)), 0)

        return temp_matrix


    def filter_coocmatrix(self):
        """
        Finds the 30% highest occurring elements from the matrix, deletes the columns and rows that do not contain
        at least one of this element. Rebuild the matrix storing on the variable "filtered matrix"

        Returns
        -------
        None
        """

        #  Gets the number that corresponds to 30% of the total of elements in the matrix
        thirty_percent = int(np.ceil(self.noun_rows_size * self.verb_columns_size * 0.3))

        #  Creates an empty numpy array with the size of the entire matrix
        temp_matrix_list = np.empty(self.noun_rows_size * self.verb_columns_size)

        x = 0

        #  Fit the matrix in a 1D array
        for i in self.matrix.flat:
            temp_matrix_list[x] = i
            x += 1

        #  Get a list of indices of the 30% most occurring items on the "1D matrix"
        thirty_percent_bigger_ind = heapq.nlargest(thirty_percent,
                                                   range(self.noun_rows_size * self.verb_columns_size),
                                                   temp_matrix_list.take)

        i = 0
        real_aij_matrix_pos = []

        #  transform the obtained 1D-matrix's indices to the indices of the real matrix (2D indices)
        while i < thirty_percent:
            real_aij_matrix_pos.append([thirty_percent_bigger_ind[i] // self.verb_columns_size,
                                        thirty_percent_bigger_ind[i] % self.verb_columns_size])
            i += 1

        self.filtered_verb_columns_size = 0
        self.filtered_noun_rows_size = 0
        noun_row_idxs = {}
        verb_column_idxs = {}

        #  Separates the indices in row and column and store them as key in a dictionary to avoid repetition
        for two_index in real_aij_matrix_pos:
            if two_index[0] not in noun_row_idxs:
                noun_row_idxs[two_index[0]] = 0
                self.filtered_noun_rows_size += 1
            if two_index[1] not in verb_column_idxs:
                verb_column_idxs[two_index[1]] = 0
                self.filtered_verb_columns_size += 1

        k = 0
        self.filtered_matrix = np.zeros((self.filtered_noun_rows_size, self.filtered_verb_columns_size))

        #  Invert the dictionaries, now they have the index as the key and the word name as the value
        temp_dict_noun = dict(zip(self.noun_rows.values(), self.noun_rows.keys()))
        temp_dict_verb = dict(zip(self.verb_columns.values(), self.verb_columns.keys()))

        #  This loop will build the filtered_matrix iterating over every index of the old matrix and checking if
        #  this is one that it can discard.
        for i in range(self.noun_rows_size):
            if i not in noun_row_idxs:
                continue
            else:
                #  This rebuilds the dictionaries that relate every noun with a row of the matrix
                self.filtered_noun_rows[temp_dict_noun[i]] = k
                l = 0
            for j in range(self.verb_columns_size):
                if j not in verb_column_idxs:
                    continue
                else:
                    self.filtered_matrix[k][l] = self.matrix[i][j]
                    #  This rebuilds the dictionaries that relate every verb with a column of the matrix
                    self.filtered_verb_columns[temp_dict_verb[j]] = l
                    l += 1
            k += 1


    def cosine_vector_sim(self, noun1, noun2, with_filtered_matrix = True):
        """
        This function calculates the cosine similarity between two word vectors
        :param noun1: The name of a noun that belongs to the co-occurrence matrix to be compared
        :param noun2: The name of the other noun to be compared
        :param with_filtered_matrix: Boolean that tells in which matrix the cosine will be calculated
        :return: The result of the cosine difference
        """
        row_noun1 = self.filtered_noun_rows[noun1]
        row_noun2 = self.filtered_noun_rows[noun2]

        if with_filtered_matrix:
            curr_matrix = self.filtered_matrix
        else:
            curr_matrix = self.matrix

        row1 = curr_matrix[row_noun1]
        row2 = curr_matrix[row_noun2]

        return 1 - cosine(row1, row2)


    def plot_two_word_vectors(self, noun1_name, noun2_name, verb1_name, verb2_name, with_filtered_matrix = True):

        if with_filtered_matrix:
            curr_matrix = self.filtered_matrix
            curr_noun_dict = self.filtered_noun_rows
            curr_verb_dict = self.filtered_verb_columns
        else:
            curr_matrix = self.matrix
            curr_noun_dict = self.noun_rows
            curr_verb_dict = self.verb_columns

        print(curr_noun_dict)
        print(curr_verb_dict)

        try:
            vector_1 = [curr_matrix[curr_noun_dict[noun1_name]][curr_verb_dict[verb1_name]],
                        curr_matrix[curr_noun_dict[noun1_name]][curr_verb_dict[verb2_name]]]

            vector_2 = [curr_matrix[curr_noun_dict[noun2_name]][curr_verb_dict[verb1_name]],
                        curr_matrix[curr_noun_dict[noun2_name]][curr_verb_dict[verb2_name]]]
        except KeyError:
            return False

        utils.plot_vectors(vector_1, vector_2, noun1_name, noun2_name, verb1_name, verb2_name)
        return True

    def create_soc_pmi_matrix(self, matrix):
        if self.is_pmi_calculated is False:
            print('The matrices have to have their pmi values computed before calling this method ')
            return

        max_value = -1

        # Noun per Noun matrix
        self.soc_pmi_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))

        # Default values, after they will need some calculation to be effective
        delta = 0.9
        gama = 3

        beta_constant = np.log2(matrix.shape[1])/delta

        i = 0
        while i < self.soc_pmi_matrix.shape[0] - 1:
            beta1 = np.maximum(int(np.floor(calc_beta(matrix, beta_constant, i))), 1)
            # print('beta1 = ' + str(beta1))
            # print('matrix_dim = ' + str(matrix.shape[0]))
            counts = zero_counter(matrix[i])
            zero_counts1 = counts

            beta_1_divisor = np.maximum((beta1-zero_counts1), 1)

            beta_largest_pmi1 = heapq.nlargest(beta1, range(matrix.shape[1]), matrix[i].take)

            j = i + 1
            while j < self.soc_pmi_matrix.shape[0]:
                beta2 = np.maximum(int(np.floor(calc_beta(matrix, beta_constant, j))), 1)

                counts = zero_counter(matrix[j])
                zero_counts2 = counts

                beta_largest_pmi2 = heapq.nlargest(beta2, range(matrix.shape[1]),
                                                   matrix[j].take)

                common_indices = np.intersect1d(beta_largest_pmi1, beta_largest_pmi2)

                f_beta1 = f_beta2 = 0
                for index in common_indices:
                    f_beta1 += np.power(matrix[j][index], gama)
                    f_beta2 += np.power(matrix[i][index], gama)

                self.soc_pmi_matrix[i][j] = (f_beta1/beta_1_divisor) + (f_beta2/np.maximum((beta2-zero_counts2), 1))
                # if max_value < self.soc_pmi_matrix[i][j]:
                #     max_value = self.soc_pmi_matrix[i][j]
                #     print('max_value = ' + str(max_value) + ' | i = ' + str(i) + ' | j = ' + str(j))


                j += 1

            i += 1
            print(i)

        # self.soc_pmi_matrix /= max_value
        # print(max_value)

        print('end')


def calc_beta(matrix, beta_constant, row_index):
    # print('np.power(np.log(matrix[row_index].sum()) ,2) = ' + str(np.power(np.log(matrix[row_index].sum()) ,2)))
    # print('np.log2(matrix.shape[1])/delta = ' + str(np.log2(matrix.shape[1])/delta))
    # print('beta = ' + str(np.power(np.log(matrix[row_index].sum()) ,2) * np.log2(matrix.shape[1])/delta))
    return np.power(np.log(matrix[row_index].sum()), 2) * beta_constant


def zero_counter(vec):
    counter = 0
    for element in vec:
        if element == 0:
            counter += 1

    return counter
