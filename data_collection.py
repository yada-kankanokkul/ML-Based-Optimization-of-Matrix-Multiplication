import os
import numpy as np
import pandas as pd
import time
import memory_profiler  # If this doesn't work, try switching to psutil
import matplotlib.pyplot as plt
from scipy.linalg import blas

# Function to generate dense matrix
def generate_dense_matrix(size):
    return np.random.rand(size, size)

# Function to generate sparse matrix (10% non-zero entries)
def generate_sparse_matrix(size):
    sparse_matrix = np.random.rand(size, size)
    sparse_matrix[sparse_matrix < 0.9] = 0  # 90% zeros
    return sparse_matrix

# Naive matrix multiplication (O(n³))
def naive_multiplication(A, B):
    return np.dot(A, B)

# Measure execution time and memory usage
def benchmark_matrix_multiplication(A, B, algorithm):
    # Measure time
    start_time = time.time()
    
    # Track memory usage using memory_profiler
    mem_usage_before = memory_profiler.memory_usage()[0]
    
    result = algorithm(A, B)
    
    end_time = time.time()
    mem_usage_after = memory_profiler.memory_usage()[0]
    
    exec_time = end_time - start_time
    mem_usage = mem_usage_after - mem_usage_before  # Memory difference in MB
    
    return exec_time, mem_usage, result


def printout(df, filename='matrix_benchmark_plot.png'):
    # Create 'results' directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Plot the execution time and memory usage for different matrix sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot execution time
    for matrix_type in df['Type'].unique():
        subset = df[df['Type'] == matrix_type]
        ax1.plot(subset['Size'], subset['Time (s)'], label=matrix_type)
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Matrix Multiplication Execution Time')
    ax1.legend()

    # Plot memory usage
    for matrix_type in df['Type'].unique():
        subset = df[df['Type'] == matrix_type]
        ax2.plot(subset['Size'], subset['Memory (MB)'], label=matrix_type)
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Matrix Multiplication Memory Usage')
    ax2.legend()

    plt.tight_layout()

    # Save the plot to an image file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    save_path = os.path.join(os.getcwd(), 'results', filename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}.")  # Debugging statement
    plt.show()


def save_to_csv(df, filename='matrix_benchmark_results.csv'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    save_path = os.path.join(os.getcwd(), 'results', filename)
    try:
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}.")  # Debugging statement
    except Exception as e:
        print(f"Error saving to CSV: {e}")  # Error handling


def main():
    # dense_100x100 = generate_dense_matrix(100)
    # dense_500x500 = generate_dense_matrix(500)
    # sparse_100x100 = generate_sparse_matrix(100)
    # sparse_500x500 = generate_sparse_matrix(500)

    # Run benchmarks for various matrices and algorithms
    matrix_sizes = [100, 500, 1000]  # Sizes: 100x100, 500x500, 1000x1000

    results = []

    for size in matrix_sizes:
        # Generate dense and sparse matrices
        dense_matrix = generate_dense_matrix(size)
        sparse_matrix = generate_sparse_matrix(size)
        
        # Benchmark dense x dense matrix multiplication
        exec_time, mem_usage, result = benchmark_matrix_multiplication(dense_matrix, dense_matrix, naive_multiplication)
        results.append({'Size': size, 'Type': 'Dense', 'Algorithm': 'Naive', 'Time (s)': exec_time, 'Memory (MB)': mem_usage})
        
        # Benchmark sparse x sparse matrix multiplication
        exec_time, mem_usage, result = benchmark_matrix_multiplication(sparse_matrix, sparse_matrix, naive_multiplication)
        results.append({'Size': size, 'Type': 'Sparse', 'Algorithm': 'Naive', 'Time (s)': exec_time, 'Memory (MB)': mem_usage})

         # Benchmark dense x sparse matrix multiplication
        exec_time, mem_usage, result = benchmark_matrix_multiplication(dense_matrix, sparse_matrix, naive_multiplication)
        results.append({'Size': size, 'Type': 'Dense x Sparse', 'Algorithm': 'Naive', 'Time (s)': exec_time, 'Memory (MB)': mem_usage})
        
        # Benchmark sparse x dense matrix multiplication
        exec_time, mem_usage, result = benchmark_matrix_multiplication(sparse_matrix, dense_matrix, naive_multiplication)
        results.append({'Size': size, 'Type': 'Sparse x Dense', 'Algorithm': 'Naive', 'Time (s)': exec_time, 'Memory (MB)': mem_usage})


    # Convert results to DataFrame
    df = pd.DataFrame(results)
    print(df.head())  # Display the first few rows
    printout(df)
    save_to_csv(df)

if __name__ == "__main__":
    main()
