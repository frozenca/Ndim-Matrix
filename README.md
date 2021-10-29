# Ndim-Matrix
C++20 N-dimensional Matrix class for hobby project

# Supporting matrix operations
- STL compatible iterators
- reshape (O(1) move operation, no copy)
- submatrix (implemented as a non-owning view, no copy)
- row, col (no copy, basically non-owning views)
- transpose
- add, sub, mul, div, modulus (supports broadcasting)
- scalar assignment, addition, subtraction, multiplication, division, etc (operations defined provided that scalar types are compatible)
- ```operator+=, -=, *=, /=, %=```, etc

# Supporting linear algebra operations
- Inner product
- Matrix multiplication
- Outer product
- LUP decomposition + Gaussian elimination
- QR decomposition
- Deteminant, Trace
- Norm
- Vector normalization
- Singular Value Decomposition (able to low rank approx SVD)
- Matrix Inverse, Pseudoinverse
- Eigenvalues
- Eigenvalues + Eigenvectors


# ToDoList
- File I/O
- Einsum
- Coordinatewise initialization
- Sparse matrix
- FFT
- Matrix over discrete scalars (In particular, Z_p)
- Array manipulations like concat, split, row/column rotate, zeropad, etc (tedious...)
- Expand linear algebra stuffs to >2D (tedious...)
- Neural network operations (CNN, RNN, etc)
- N-dim sampling (N-dim Gaussian, t-Student, Gamma, Wishart, etc)
- Migrate to C++20 modules (Current MSVC 19.28 implementation is broken, waiting for fix)
- Binding with Python
- Documentation for APIs
- Build testing frameworks
