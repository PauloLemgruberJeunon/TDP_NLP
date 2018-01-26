import numpy as np

a = np.array([[1,2,3],[4,5,6],[0,0,0],[5,2,3],[0,0,1],[0,0,0]])

print(a)

i = 0
while i < a.shape[0]:
    if np.sum(a[i]) == 0:
        a = np.delete(a,i,0)
        i -= 1

    i += 1

print(a)