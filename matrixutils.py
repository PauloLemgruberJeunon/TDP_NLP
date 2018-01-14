import numpy as np
import heapq

class CoocMatrix:
    """
    Class responsible to handle the co-occurrence matrix operations
    """

    def __init__(self, windows):
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

        #  Functions called to calculate the co-occurrence matrix and it's filtered version
        self.create_coocmatrix(windows)
        self.filter_coocmatrix()


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

        window_verbs = []
        window_nouns = []

        for window in windows:  # Iterates over the windows
            for (word, tag) in window: # Iterates over each tagged word inside the window
                if tag.startswith('V') is True:  # If it is a verb store it on a dictionary and increase
                    window_verbs.append(word)                                                #  the size of columns
                    if word not in self.verb_columns:
                        self.verb_columns[word] = self.verb_columns_size
                        self.verb_columns_size += 1
                        if self.verb_columns_size > 2:
                            self.matrix = np.lib.pad(self.matrix, ((0, 0), (0, 1)), 'constant', constant_values=0)

                if tag.startswith('NN') is True:  # If it is a Noun do the same as the verbs, but increase the rows
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

            window_verbs.clear() # Clear temp lists for next iteration
            window_nouns.clear()


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

        new_verb_columns = {}
        new_noun_rows = {}
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
                new_noun_rows[temp_dict_noun[i]] = k
                l = 0
            for j in range(self.verb_columns_size):
                if j not in verb_column_idxs:
                    continue
                else:
                    self.filtered_matrix[k][l] = self.matrix[i][j]
                    #  This rebuilds the dictionaries that relate every verb with a column of the matrix
                    new_verb_columns[temp_dict_verb[j]] = l
                    l += 1
            k += 1
