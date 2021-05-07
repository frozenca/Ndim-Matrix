# Ndim-Matrix
C++20 N-dimensional Matrix class for hobby project

# Supports
reshape (O(1) move operation, no copy)
submatrix (implemented as a non-owning view, no copy)
row, col (no copy, basically non-owning views)
add, sub, mul, div, modulus (supports broadcasting)
scalar assignment, addition, subtraction, multiplication, division, etc (operations defined provided that scalar types are compatible)
operator+=, -=, *=, /=, %=, etc

# Supports 
Inner product
Matrix multiplication
Outer product
LUP decomposition + Gaussian elimination
QR decomposition
Deteminant, Trace
Norm
Vector normalization
Singular Value Decomposition (able to low rank approx SVD)
Matrix Inverse
Eigenvalues
Eigenvalues + Eigenvectors


# ToDoList
Array manipulations like concat, split, etc (tedious...)
Expand linear algebra stuffs to >2D (tedious...)
Neural network operations (CNN, RNN, etc)
N-dim sampling (N-dim Gaussian, etc)
Migrate to C++20 modules (Current MSVC 19.28 implementation is broken, waiting for fix)
Build testing frameworks
