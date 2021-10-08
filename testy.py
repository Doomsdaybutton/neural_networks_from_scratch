import numpy as np

# testing numpy matrix, vector, ... stuff
print("Numpy Cheat-Sheet\n-----------------------------------------\n-----------------------------------------\n\n\n")
np.random.seed(0)


matrix = np.random.randn(3, 3)
print("""
-----------------------------------------
Matrix Transpose

matrix =    [[a, b, c],
             [d, e, f],
             [g, h, i]]

matrix.T=   [[a, d, g],
             [b, e, h],
             [c, f, i]]

With random values:

matrix = 
{0}

matrix.T =
{1}
-----------------------------------------

""".format(matrix, matrix.T))

# dot product

# two vectors
v1 = np.array([2, 3, 4])
v2 = np.array([3, 1, 2])

print("""
-----------------------------------------
Dot product of two vectors:

v1 = {0}
v2 = {1}
np.dot(v1, v2) = {2} = 2*3 + 3*1 + 4*2
-----------------------------------------

""".format(v1, v2, np.dot(v1, v2)))

## vector and matrix (vector-matrix-multiplication)

print("""
-----------------------------------------
Dot product of vector and matrix (vector-matrix-multiplication)

v1 = {0}
matrix = {1}

np.dot(matrix, v1) = [np.dot(matrix[0], v1), np.dot(matrix[1], v1), ...] = {2}
-----------------------------------------

""".format(v1, matrix, np.dot(matrix, v1)))

## vectors and matrices

m = np.random.randn(3, 4)
vs = np.random.randn(4, 2)

print("""
-----------------------------------------
Matrix-vectors-multiplication

matrix =
[[a, b, c, d],
 [e, f, g, h],
 [i, j, k, l]]

vectors =
[[m, n],
 [o, p],
 [q, r],
 [s, t]]

np.dot(matrix, Vectors):
[[am+bo+cq+ds, an+bp+cr+dt],
 [em+fo+gq+hs, en+fp+gr+ht],
 [im+jo+kq+ls, in+ip+ir+it]]

With random values:

matrix:
{0}

vectors:
{1}

np.dot(matrix, vectors):
{2}
-----------------------------------------

""".format(m, vs, np.dot(m, vs)))

print("""
-----------------------------------------
The dot product:

v1 = {0}
v2 = {1}
v1*v2 = {2}

np.sum([5, 6, 2]) = {3}
np.sum(v1*v2) = {4}
np.dot(v1, v2) = {5}
-----------------------------------------

""".format(v1, v2, v1*v2, np.sum([5, 6, 2]), np.sum(v1*v2), np.dot(v1, v2)))

print("""
-----------------------------------------
np.sum:

v1 = {0}

np.sum(v1) = {1}

matrix = {2}

np.sum(matrix) = {3}

np.sum(matrix, axis=0) = {4}

np.sum(matrix, axis=1) = {5}

np.sum(matrix, axis=1, keepdims=True) = {6}
-----------------------------------------

""".format(v1, np.sum(v1), matrix, np.sum(matrix), np.sum(matrix, axis=0), np.sum(matrix, axis=1), np.sum(matrix, axis=1, keepdims=True)))

m1 = np.random.randn(2, 3)
m2 = np.random.randn(2, 3)
print("""
-----------------------------------------
multiplication of matrices (not dot product):

matrix1 = {0}

matrix2 = {1}

matrix1*matrix2 = {2}
-----------------------------------------

""".format(m1, m2, m1*m2))

mm1 = np.random.randn(4, 5)
print("""
-----------------------------------------
argmax:

matrix = {0}

np.argmax(matrix) = {1}

np.argmax(matrix, axis=0) = {2}

np.argmax(matrix, axis=1) = {3}
-----------------------------------------

""".format(mm1, np.argmax(mm1), np.argmax(mm1, axis=0), np.argmax(mm1, axis=1)))

# batching
# mylist = np.zeros(100)
# for i in range(mylist.size):
#     mylist[i] = i

# print(mylist)
# for i in range(0, mylist.size, 15):
#     minilist = mylist[i:i+15]
#     print(minilist)

# gh = np.random.randn(5, 7)
# print(len(gh))

# import json
# import os
# config_file = open('//stu.net.fr.ch/perso$/Users/ProencaM/Documents/Data/MA/code/config.json')
# config = json.load(config_file)
# config_file.close()
# print(config['path_to_training_data'])
print(__file__.replace('\\', '/')[:__file__.find('test')])
