import os
import numpy as np
import time
import pickle
from Matrix_Multiplication import naive_multiplication, block_multiplication, strassen_multiplication
from Parallel_Multiplication import parallel_multiplication
from Matrix_generation import generate_dense_matrix, generate_sparse_matrix

def save_data(matrix, filename, folder="data"):
    """Save a matrix to a file."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, filename), 'wb') as f:
        pickle.dump(matrix, f)


def load_data(filename, folder="data"):
    """Load a matrix from a file."""
    with open(os.path.join(folder, filename), 'rb') as f:
        return pickle.load(f)
    

def execute_multiplication(A, B, method, parallel=False, num_processes=1):
    """Perform matrix multiplication using the specified method."""
    start_time = time.time()

    if method == 'naive':
        if parallel:
            result = parallel_multiplication(A, B, method='naive', num_processes=num_processes)
        else:
            result = naive_multiplication(A, B)
    elif method == 'strassen':
        if parallel:
            result = parallel_multiplication(A, B, method='strassen', num_processes=num_processes)
        else:
            result = strassen_multiplication(A, B)
    elif method == 'block':
        if parallel:
            result = parallel_multiplication(A, B, method='block', num_processes=num_processes)
        else:
            result = block_multiplication(A, B)
    else:
        raise ValueError(f"Unknown method: {method}")

    exec_time = time.time() - start_time
    mem_usage = A.nbytes + B.nbytes + result.nbytes

    return result, exec_time, mem_usage

def save_results(results, filename, folder="results"):
    """Save multiplication results to a file."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, filename), 'wb') as f:
        pickle.dump(results, f)

def main():
    matrix_sizes = [64, 1024, 4096]
    methods = ['naive', 'strassen', 'block']
    parallel_methods = ['naive_parallel', 'strassen_parallel', 'block_parallel']
    sample_size = 1

    # Generate matrices and save them
    for size in matrix_sizes:
        for i in range(sample_size):  
            dense_matrix = generate_dense_matrix(size)
            sparse_matrix = generate_sparse_matrix(size)
            save_data(dense_matrix, f"dense_{size}_{i}.pkl")
            save_data(sparse_matrix, f"sparse_{size}_{i}.pkl")

    # Perform all combinations of multiplication and record results
    results = []
    for size in matrix_sizes:
        for i in range(sample_size):
            for j in range(sample_size):
                dense_A = load_data(f"dense_{size}_{i}.pkl")
                dense_B = load_data(f"dense_{size}_{j}.pkl")
                sparse_A = load_data(f"sparse_{size}_{i}.pkl")
                sparse_B = load_data(f"sparse_{size}_{j}.pkl")

                # Combinations: Sparse x Sparse, Dense x Dense, Dense x Sparse, Sparse x Dense
                matrix_combinations = [
                    (sparse_A, sparse_B, "Sparse x Sparse"),
                    (dense_A, dense_B, "Dense x Dense"),
                    (dense_A, sparse_B, "Dense x Sparse"),
                    (sparse_A, dense_B, "Sparse x Dense"),
                ]

                for A, B, description in matrix_combinations:
                    for method in methods + parallel_methods:
                        parallel = 'parallel' in method
                        base_method = method.replace('_parallel', '')

                        try:
                            result, exec_time, mem_usage = execute_multiplication(
                                A, B, base_method, parallel=parallel, num_processes=4
                            )
                            results.append({
                                'description': description,
                                'method': method,
                                'size': size,
                                'exec_time': exec_time,
                                'mem_usage': mem_usage,
                            })
                        except Exception as e:
                            print(f"Failed for {description} using {method}: {e}")

    # Save results
    save_results(results, "multiplication_results.pkl")

    # Reinforcement Learning logic placeholder
    print("Starting reinforcement learning to choose the best method...")
    # Implement RL logic here
    # Example: best_method = reinforcement_learning(results)


if __name__ == "__main__":
    main()