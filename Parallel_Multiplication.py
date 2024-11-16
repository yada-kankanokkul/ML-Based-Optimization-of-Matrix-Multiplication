import numpy as np
from concurrent.futures import ThreadPoolExecutor
from Matrix_Multiplication import naive_multiplication, strassen_multiplication, block_multiplication

def parallel_multiplication(A, B, method='naive', num_threads=4, block_size=2):
    """
    Perform matrix multiplication in parallel using the specified method.

    Parameters:
    A, B: np.ndarray
        Input matrices to multiply.
    method: str
        The method to use for matrix multiplication. Options: 'naive', 'strassen', 'block'.
    num_threads: int
        Number of threads to use for parallel computation.
    block_size: int
        Block size for block multiplication (used only if method is 'block').

    Returns:
    np.ndarray
        Result of the matrix multiplication.
    """
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    # Ensure valid matrix multiplication
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must equal the number of rows in B.")

    # Initialize result matrix
    result = np.zeros((rows_A, cols_B))

    # Select the multiplication method
    if method == 'naive':
        multiplication_func = naive_multiplication
    elif method == 'strassen':
        multiplication_func = strassen_multiplication
    elif method == 'block':
        multiplication_func = lambda x, y: block_multiplication(x, y, block_size=block_size)
    else:
        raise ValueError("Invalid method. Choose from 'naive', 'strassen', or 'block'.")

    def compute_row(row_index):
        """Compute a single row of the result matrix."""
        for col_index in range(cols_B):
            # Slice rows and columns for block multiplication
            result[row_index, col_index] = multiplication_func(
                A[row_index:row_index + 1, :],
                B[:, col_index:col_index + 1]
            )[0, 0]

    # Perform parallel computation
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(compute_row, range(rows_A))

    return result


if __name__ == "__main__":
    # Input matrices
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)

    # Perform parallel multiplication with naive method
    print("Naive Multiplication in Parallel:")
    result_naive = parallel_multiplication(A, B, method='naive', num_threads=4)
    print(result_naive)

    # Perform parallel multiplication with strassen method
    print("\nStrassen Multiplication in Parallel:")
    result_strassen = parallel_multiplication(A, B, method='strassen', num_threads=4)
    print(result_strassen)

    # Perform parallel multiplication with block method
    print("\nBlock Multiplication in Parallel:")
    result_block = parallel_multiplication(A, B, method='block', num_threads=4, block_size=2)
    print(result_block)
