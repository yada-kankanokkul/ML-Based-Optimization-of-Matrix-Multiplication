import numpy as np

# Function to generate dense matrix
def generate_dense_matrix(size):
    return np.random.rand(size, size)

# Function to generate sparse matrix (10% non-zero entries by defult)
def generate_sparse_matrix(size, sparsity_percentage=0.9):
    sparse_matrix = np.random.rand(size, size)
    sparse_matrix[sparse_matrix < sparsity_percentage] = 0
    return sparse_matrix


if __name__ == "__main__":
    size = 8  # Matrix size
    sparsity = 0.9  # Adjust sparsity percentage
    matrix = generate_sparse_matrix(size, sparsity)
    print("Generated Sparse Matrix:")
    print(matrix)