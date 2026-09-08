# Arrays, shape, dtype and indexing
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)

# Create a numpy array
np.zeros(2)
np.ones(2)
## np array with n random elements
np.empty(2)
## np array with n ordered elements
np.arange(2)
## Separate a linearspace into n evenly spaced elements
np.linspace(0, 10, num=5)

# Sorting
np.sort(a)
# Concatenate numpy arrays
np.concatenate((a, b), axis=0)
# Reshape a numpy array
a.reshape(3, 2)
np.reshape(a, (3, 2))
# Add a new axis/dimension
np.newaxis
np.expand_dims(a, axis=0)
# Indexing and slicing
a[0][1]
a[0][:2]
# Filtering and boolean array
a[a >= 4]
a >= 4

# Create a numpy array from existing data
a1 = np.array([[1, 1], [2, 2]])
a2 = np.array([[3, 3], [4, 4]])
## Vertically and horizontally stack
np.vstack((a1, a2))
np.hstack((a1, a2))
## Vertically (row-wise) and horizontally (column-wise) split
np.vsplit(a, 2)
np.hsplit(a, 2) 
## View (shallow copy; would modify data in source array) 
a.view()
## Copy (complete copy of data; won't affect source array)
a.copy()

# Broadcasting (a dimension of two arrays are either equal or one of them is 1)
np.array([[1, 2, 3], [4, 5, 6]]) + np.array([7, 8, 9])
# Linear algebra (np.linalg)
np.dot()
np.matmul() or @
np.transpose() or .T
