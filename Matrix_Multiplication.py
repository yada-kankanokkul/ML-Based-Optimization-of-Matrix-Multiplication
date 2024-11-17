import numpy as np

# Check matric size - Ensure the matrices can be multiplied
def isValidSize(A, B):
    cols_A = A.shape[1]
    rows_B = B.shape[0]
    if cols_A == rows_B:
        return True
    else:
        return False
        
def naive_multiplication(A, B):
    """
    Perform matrix multiplication using the naive approach.

    Parameters:
    A, B: np.ndarray
        Input matrices.

    Returns:
    np.ndarray
        Result of the multiplication A x B.
    """
    # Ensure the matrices can be multiplied
    if not isValidSize(A, B):
        raise ValueError("Number of columns in A must equal the number of rows in B.")
    
    # Get the dimensions of the input matrices
    m, inner  = A.shape
    n = B.shape[1]
    
    # Initialize the result matrix with zeros
    result = np.zeros((m, n))

    # Perform the multiplication
    for i in range(m):
        for j in range(n):
            for k in range(inner):
                result[i][j] += A[i][k] * B[k][j]

    return result


def block_multiplication(A, B, block_size=2):
    """
    Perform matrix multiplication using the block method.

    Parameters:
    A, B: np.ndarray
        Input matrices.
    block_size: int
        Size of the square blocks.

    Returns:
    np.ndarray
        Result of the multiplication A x B which is [mxn] size.
    """
    # Ensure the matrices can be multiplied
    if not isValidSize(A, B):
        raise ValueError("Number of columns in A must equal the number of rows in B.")
    
    # Get the dimensions of the input matrices
    m, inner  = A.shape
    n = B.shape[1]

    # Initialize the result matrix with zeros
    result = np.zeros((m, n))

    # Divide matrices into blocks and compute partial products
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            for k in range(0, inner, block_size):
                # Define blocks
                A_block = A[i : i+block_size, k : k+block_size]
                B_block = B[k : k+block_size, j : j+block_size]

                # Add the product of blocks to the result
                result[i : i+block_size, j : j+block_size] += np.dot(A_block, B_block)

    return result

# strassen_multiplication - Utility to check if a number is a power of two
def is_power_of_two(n):
    """Check if a number is a power of two."""
    return (n > 0) and (n & (n - 1)) == 0

def strassen_multiplication(A, B):
    """
    Perform matrix multiplication using Strassen's algorithm.

    Parameters:
    A, B: np.ndarray
        Square matrices of the same dimension (must be 2^n x 2^n).
    pool:
        Use in case of parallel.

    Returns:
    np.ndarray
        Result of the multiplication A x B.
    """


    # Ensure matrices are square and dimensions are 2^n
    if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
        raise ValueError("Matrices must be square.")
    
    if not is_power_of_two(A.shape[0]) or not is_power_of_two(B.shape[0]):
        raise ValueError("Matrix dimensions must be powers of 2.")
    
    # Base case: when the matrix size is 1x1
    if A.shape[0] == 1:
        return A * B

    # Split the matrices into quadrants
    mid = A.shape[0] // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]


    # Compute the 7 products (Strassen's algorithm key step)
    M1 = strassen_multiplication(A11 + A22, B11 + B22)
    M2 = strassen_multiplication(A21 + A22, B11)
    M3 = strassen_multiplication(A11, B12 - B22)
    M4 = strassen_multiplication(A22, B21 - B11)
    M5 = strassen_multiplication(A11 + A12, B22)
    M6 = strassen_multiplication(A21 - A11, B11 + B12)
    M7 = strassen_multiplication(A12 - A22, B21 + B22)


    # Combine the results into C's quadrants
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Combine quadrants into a single matrix
    C = np.zeros((A.shape[0], A.shape[1]))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

    return C



# # strassen_multiplication - Utility to check if a number is a power of two
# def is_power_of_two(n):
#     """Check if a number is a power of two."""
#     return (n > 0) and (n & (n - 1)) == 0

# def strassen_multiplication(A, B, pool=None):
#     """
#     Perform matrix multiplication using Strassen's algorithm.

#     Parameters:
#     A, B: np.ndarray
#         Square matrices of the same dimension (must be 2^n x 2^n).
#     pool:
#         Use in case of parallel.

#     Returns:
#     np.ndarray
#         Result of the multiplication A x B.
#     """


#     # Ensure matrices are square and dimensions are 2^n
#     if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
#         raise ValueError("Matrices must be square.")
    
#     if not is_power_of_two(A.shape[0]) or not is_power_of_two(B.shape[0]):
#         raise ValueError("Matrix dimensions must be powers of 2.")
    
#     # Base case: when the matrix size is 1x1
#     if A.shape[0] == 1:
#         return A * B

#     # Split the matrices into quadrants
#     mid = A.shape[0] // 2
#     A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
#     B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]

#     if pool is None:
#         # Compute the 7 products (Strassen's algorithm key step)
#         M1 = strassen_multiplication(A11 + A22, B11 + B22)
#         M2 = strassen_multiplication(A21 + A22, B11)
#         M3 = strassen_multiplication(A11, B12 - B22)
#         M4 = strassen_multiplication(A22, B21 - B11)
#         M5 = strassen_multiplication(A11 + A12, B22)
#         M6 = strassen_multiplication(A21 - A11, B11 + B12)
#         M7 = strassen_multiplication(A12 - A22, B21 + B22)
#     else:
#         # Parallel approach
#         results = pool.starmap(strassen_multiplication, [
#             (A11 + A22, B11 + B22, pool),
#             (A21 + A22, B11, pool),
#             (A11, B12 - B22, pool),
#             (A22, B21 - B11, pool),
#             (A11 + A12, B22, pool),
#             (A21 - A11, B11 + B12, pool),
#             (A12 - A22, B21 + B22, pool),
#         ])
#         # Extract the results from the pool
#         M1, M2, M3, M4, M5, M6, M7 = results

#     # Combine the results into C's quadrants
#     C11 = M1 + M4 - M5 + M7
#     C12 = M3 + M5
#     C21 = M2 + M4
#     C22 = M1 - M2 + M3 + M6

#     # Combine quadrants into a single matrix
#     C = np.zeros((A.shape[0], A.shape[1]))
#     C[:mid, :mid] = C11
#     C[:mid, mid:] = C12
#     C[mid:, :mid] = C21
#     C[mid:, mid:] = C22

#     return C





# Example Usage:
if __name__ == "__main__":
    A = np.random.rand(4, 4)  # 8x8 matrix
    B = np.random.rand(4, 4)  # 8x8 matrix

    # C = np.array([[1, 2, 3],
    #               [4, 5, 6]])

    # D = np.array([[7, 8],
    #               [9, 10],
    #               [11, 12]])

    result1 = strassen_multiplication(A, B)
    print("Result of Strassen's Multiplication:")
    print(result1)

    result2 = naive_multiplication(A, B)
    print("Result of Naive Multiplication:")
    print(result2)

  
    result3 = block_multiplication(A, B, 2)
    print("Result of Block Multiplication:")
    print(result3)
