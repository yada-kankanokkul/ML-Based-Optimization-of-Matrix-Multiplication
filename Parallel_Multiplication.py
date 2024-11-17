import numpy as np
import multiprocessing

# Function to check if two matrices can be multiplied
def isValidSize(A, B):
    cols_A = A.shape[1]
    rows_B = B.shape[0]
    return cols_A == rows_B

# Function to check if a number is a power of two
def is_power_of_two(n):
    if n <= 0:
        return False
    return (n & (n - 1)) == 0

# Worker function for naive multiplication
# This function handles the matrix multiplication for a specific subset of rows
def naive_worker(args):
    A, B, row_range = args # Unpack arguments: matrix A, matrix B, and the subset of rows to process

    rows = len(row_range)
    cols = B.shape[1]
    result = np.zeros((rows, cols)) # Initialize zero matrix

    for idx, i in enumerate(row_range):
        for j in range(cols):
            for k in range(A.shape[1]):
                result[idx][j] += A[i][k] * B[k][j]

    return result

# Naive multiplication using parallel processing
def naive_multiplication(A, B, pool, num_processes):
    rows = A.shape[0]
    row_ranges = np.array_split(range(rows), num_processes)
    tasks = []
    for rows in row_ranges:
        task = (A, B, rows)
        tasks.append(task)

    results = pool.map(naive_worker, tasks)

    return np.vstack(results)

# Worker function for block multiplication
def block_worker(args):
    A_block, B_block = args
    return np.dot(A_block, B_block)

# Block multiplication using parallel processing
def block_multiplication(A, B, pool, block_size):
    rows = A.shape[0]
    cols = B.shape[1]
    inner = A.shape[1]
    result = np.zeros((rows, cols))

    # Divide the matrices into smaller blocks and create tasks for each block multiplication
    tasks = []
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            for k in range(0, inner, block_size):
                A_block = A[i:i + block_size, k:k + block_size]
                B_block = B[k:k + block_size, j:j + block_size]
                tasks.append((A_block, B_block))

    # Use a pool of workers to process the tasks in parallel
    results = pool.map(block_worker, tasks)

    # Reconstruct the result matrix from the computed blocks
    idx = 0
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            result[i:i + block_size, j:j + block_size] += results[idx]
            idx += 1

    return result

# Strassen multiplication function
def strassen_multiplication(A, B):
    # Strassen's algorithm is only applicable to square matrices with dimensions as powers of two
    if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
        raise ValueError("Matrices must be square.")
    if not is_power_of_two(A.shape[0]):
        raise ValueError("Matrix dimensions must be powers of 2.")

    # Base case: 1x1 matrices
    if A.shape[0] == 1:
        return A * B

    # Split matrices into quadrants
    mid = A.shape[0] // 2 # Find the midpoint to divide the matrices into quadrants
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    # Compute the 7 products required by Strassen's algorithm
    M1 = strassen_multiplication(A11 + A22, B11 + B22)
    M2 = strassen_multiplication(A21 + A22, B11)
    M3 = strassen_multiplication(A11, B12 - B22)
    M4 = strassen_multiplication(A22, B21 - B11)
    M5 = strassen_multiplication(A11 + A12, B22)
    M6 = strassen_multiplication(A21 - A11, B11 + B12)
    M7 = strassen_multiplication(A12 - A22, B21 + B22)

    # Compute the four quadrants of the resulting matrix
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Combine the quadrants into the final result
    C = np.zeros((A.shape[0], A.shape[1]))
    mid = A.shape[0] // 2
    C[:mid, :mid] = C11 # Top-left quadrant
    C[:mid, mid:] = C12 # Top-right quadrant
    C[mid:, :mid] = C21 # Bottom-left quadrant
    C[mid:, mid:] = C22 # Bottom-right quadrant

    return C

# Parallel multiplication function to choose the multiplication method
def parallel_multiplication(A, B, method='naive', num_processes=2, block_size=2):
    # Check if the matrices can be multiplied
    if not isValidSize(A, B):
        raise ValueError("Number of columns in A must equal the number of rows in B.")

    # Use multiprocessing to perform the chosen multiplication method
    with multiprocessing.Pool(processes=num_processes) as pool:
        if method == 'naive':
            return naive_multiplication(A, B, pool, num_processes)
        elif method == 'block':
            return block_multiplication(A, B, pool, block_size)
        elif method == 'strassen':
            return strassen_multiplication(A, B)
        else:
            raise ValueError(f"Unknown method '{method}'.")

# Main section
if __name__ == "__main__":
    # Generate two random 8x8 matrices for multiplication
    A = np.random.rand(8, 8)
    B = np.random.rand(8, 8)

    print("Naive Multiplication in Parallel:")
    try:
        result_naive = parallel_multiplication(A, B, method='naive', num_processes=2, block_size=2)
        print(result_naive)
    except Exception as e:
        print(f"Failed to perform naive parallel multiplication: {e}")

    print("\nStrassen Multiplication in Parallel:")
    try:
        result_strassen = parallel_multiplication(A, B, method='strassen', num_processes=2, block_size=2)
        print(result_strassen)
    except Exception as e:
        print(f"Failed to perform strassen parallel multiplication: {e}")

    print("\nBlock Multiplication in Parallel:")
    try:
        result_block = parallel_multiplication(A, B, method='block', num_processes=2, block_size=2)
        print(result_block)
    except Exception as e:
        print(f"Failed to perform block parallel multiplication: {e}")
