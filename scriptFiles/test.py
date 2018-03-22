import numpy as np

cooc_matrix = np.array([[0, 0, 1, 0, 1], [0, 0, 1, 0, 1], [2, 1, 0, 1, 0], [1, 6, 0, 4, 0]], dtype=float)
improver = 1
constant = 2

# Laplace smoothing
if improver is 1:
    temp_matrix = np.add(cooc_matrix, constant)
else:
    temp_matrix = cooc_matrix

print(temp_matrix)
print('')

# Sum all the elements in the matrix
total = temp_matrix.sum()
# Creates an array with each element containing the sum of one column
total_per_column = temp_matrix.sum(axis=0, dtype='float')
# Creates an array with each element containing the sum of one row
total_per_line = temp_matrix.sum(axis=1, dtype='float')

# print(total)
# print(total_per_column)
# print(total_per_line)

# Get the matrix dimensions
(maxi, maxj) = temp_matrix.shape

# print(maxi)
# print(maxj)
l = 0
# Iterates over all the matrix
for i in range(maxi):
    k = 0
    for j in range(maxj):

        # Calculates the PMI's constants
        pcw = temp_matrix[i][j] / total
        pw = total_per_line[i] / total
        pc = total_per_column[j] / total

        # print(pcw)
        # print(pw)
        # print(pc)

        # Checks for division per zero
        if pw * pc == 0:
            temp_matrix[i][j] = 0
        else:
            # Calculates the new wighted value
            # print(np.maximum(np.log2(pcw / (pw * pc)), 0))
            temp_matrix[l][k] = float(np.maximum(np.log2(pcw / (pw * pc)), 0))

        # input('pause')
        # print(temp_matrix)
        k+=1
    l+=1


print(temp_matrix)