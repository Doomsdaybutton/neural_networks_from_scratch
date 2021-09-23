import numpy as np

# testing numpy matrix, vector, ... stuff
print("Numpy Cheat-Sheet\n-----------------------------------------\n-----------------------------------------\n\n\n")
np.random.seed(0)

rows=4
columns=3

matrix = np.random.randn(rows, columns)

arr = np.array([5.2,4,3,2.0])

print("matrix = ")
print(matrix)
print("matrix.T = ")
print(matrix.T)

"""
matrix =    [[a, b, c],
             [d, e, f],
             [g, h, i]]

matrix.T=   [[a, d, g],
             [b, e, h],
             [c, f, i]]

"""

# dot product

## two vectors
v1 = np.array([2, 3, 4])
v2 = np.array([3, 1, 2])

v3 = np.dot(v1, v2)

print("Dot product of two vectors: ")
print(v1, v2, v3, " = 2*3+3*1+4*2")

## vector and matrix (vector-matrix-multiplication)

print("Dot product of vector and matrix (vector-matrix-multiplication)")
print(v1, "\n",matrix)
print("np.dot(matrix, v1) = np.dot(matrix[0], v1), np.dot(matrix[1], v1), ... : ")
print(np.dot(matrix, v1))

## vectors and matrices

#matrices = np.random.randn(3, 3, 4)
vectors = np.random.randn(3, 3)

#print("matrices:\n", matrices)
print("vectors:\n", vectors)

print("matrix:\n",matrix,"\nnp.dot(matrix, vectors):\n",np.dot(matrix, vectors))

print("\n\nMatrix-vectorS-multiplication:")

m=np.random.randn(3, 4)
vs=np.random.randn(4, 2)
print("matrix:\n","""
[[a, b, c, d],
 [e, f, g, h],
 [i, j, k, l]]

Vectors:
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
""".format(m, vs, np.dot(m, vs)), )

print("\n\nThe dot product:")
print("v1:", v1, "v2:", v2)
print("v1*v2", v1*v2)
print("np.sum([5, 6, 2]):", np.sum([5, 6, 2]), "np.sum(v1*v2):", np.sum(v1*v2), "np.dot(v1, v2)", np.dot(v1, v2), "\n\n")
