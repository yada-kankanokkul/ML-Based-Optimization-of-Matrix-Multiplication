import numpy as np
from multiprocessing import Pool
from Matrix_Multiplication import naive_multiplication, block_multiplication, strassen_multiplication, isValidSize


def parallel_multiplication(A, B, method='naive', num_processes=2, block_size=2):
    """
    Perform matrix multiplication in parallel using the specified method.

    Parameters:
    A, B: np.ndarray
        Input matrices.
    method: str
        Method for multiplication ('naive', 'strassen', 'block').
    num_processes: int
        Number of processes to use for parallelization.
    block_size: int
        Block size for block multiplication (ignored for other methods).

    Returns:
    np.ndarray
        Result of the matrix multiplication.
    """
    if not isValidSize(A, B):
        raise ValueError("Matrix dimensions are not compatible for multiplication.")

    def compute_element(args):
        i, j = args
        return sum(A[i, k] * B[k, j] for k in range(A.shape[1]))

    def parallel_naive():
        # Prepare arguments for each element in the result matrix
        m, n = A.shape[0], B.shape[1]
        element_indices = [(i, j) for i in range(m) for j in range(n)]
        
        # Use the pool to compute elements in parallel
        result_flat = pool.map(compute_element, element_indices)
        return np.array(result_flat).reshape(m, n)

    def parallel_block():
        # Get matrix dimensions
        m, inner = A.shape
        n = B.shape[1]

        # Prepare arguments for each block
        block_indices = [
            (i, j, k)
            for i in range(0, m, block_size)
            for j in range(0, n, block_size)
            for k in range(0, inner, block_size)
        ]

        def compute_block(args):
            i, j, k = args
            A_block = A[i:i + block_size, k:k + block_size]
            B_block = B[k:k + block_size, j:j + block_size]
            return np.dot(A_block, B_block)

        # Use the pool to compute blocks in parallel
        block_results = pool.map(compute_block, block_indices)

        # Combine results into the final matrix
        result = np.zeros((m, n))
        block_count = 0
        for i in range(0, m, block_size):
            for j in range(0, n, block_size):
                result[i:i + block_size, j:j + block_size] += block_results[block_count]
                block_count += 1

        return result

    def parallel_strassen():
        return strassen_multiplication(A, B, pool=pool)

    with Pool(num_processes) as pool:
        if method == 'naive':
            return parallel_naive()
        elif method == 'block':
            return parallel_block()
        elif method == 'strassen':
            return parallel_strassen()
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose from 'naive', 'block', or 'strassen'.")


if __name__ == "__main__":
    # Input matrices
    A = np.random.rand(8, 8)
    B = np.random.rand(8, 8)

    # Perform parallel multiplication with naive method
    print("Naive Multiplication in Parallel:")
    try:
        result_naive = parallel_multiplication(A, B, method='naive', num_processes=2)
        print(result_naive)
    except Exception as e:
        print(f"Failed to perform naive parallel multiplication: {e}")

    # Perform parallel multiplication with strassen method
    print("\nStrassen Multiplication in Parallel:")
    try:
        result_strassen = parallel_multiplication(A, B, method='strassen', num_processes=2)
        print(result_strassen)
    except Exception as e:
        print(f"Failed to perform strassen parallel multiplication: {e}")

    # Perform parallel multiplication with block method
    print("\nBlock Multiplication in Parallel:")
    try:
        result_block = parallel_multiplication(A, B, method='block', num_processes=2, block_size=2)
        print(result_block)
    except Exception as e:
        print(f"Failed to perform block parallel multiplication: {e}")
