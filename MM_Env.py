import os
import gym
from gym.spaces import Discrete, Box
import numpy as np
import time
import pickle

from Matrix_generation import generate_dense_matrix, generate_sparse_matrix
from Matrix_Multiplication import naive_multiplication, block_multiplication, strassen_multiplication
from Parallel_Multiplication import parallel_multiplication


class MatrixMultiplicationEnv(gym.Env):
    def __init__(self, 
                 matrix_size=[2**6, 2**10, 2**12],  # Matrix sizes: 64, 1024, 4096
                 multiplication_methods=['naive', 'strassen', 'block'], 
                 parallel_options=[False, True], 
                 partition_options=[2, 4], 
                 save_data=True):
        super(MatrixMultiplicationEnv, self).__init__()

        # Action Space: A combination of multiplication method, parallel option, partition option
        self.matrix_size = matrix_size
        self.multiplication_methods = multiplication_methods
        self.parallel_options = parallel_options
        self.partition_options = partition_options
        self.save_data = save_data

        # Action Space: A combination of multiplication method, parallel option, partition option
        self.action_space = Discrete(len(self.multiplication_methods) * len(self.parallel_options) * len(self.partition_options))
        self.observation_space = Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # Create directories for saving data
        self.data_dir = "data"
        self.results_dir = "results"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def reset(self):
        # Randomly choose matrix size, sparsity, and partitioning option
        self.state = np.array([np.random.choice(self.matrix_size) / max(self.matrix_size), 
                               np.random.uniform(0.1, 0.9),  # Random sparsity
                               np.random.choice(self.partition_options)])
        
        self.multiplication_method = np.random.choice(self.multiplication_methods)
        self.parallel_option = np.random.choice(self.parallel_options)
        
        # Save the matrix data for potential future visualization tasks
        if self.save_data:
            matrix_data = {'size': self.state[0], 'sparsity': self.state[1], 'partition_option': self.state[2], 'method': self.multiplication_method}
            with open(os.path.join(self.data_dir, f'matrix_data_{time.time()}.pkl'), 'wb') as f:
                pickle.dump(matrix_data, f)
        
        return self.state

    
    def step(self, action):
        # Decode action into multiplication method, parallel option, and partitioning
        method_idx = action % len(self.multiplication_methods)
        parallel_idx = (action // len(self.multiplication_methods)) % len(self.parallel_options)
        partition_idx = (action // (len(self.multiplication_methods) * len(self.parallel_options))) % len(self.partition_options)
        
        method = self.multiplication_methods[method_idx]
        parallel = self.parallel_options[parallel_idx]
        partition = self.partition_options[partition_idx]

        # Generate two matrices for multiplication based on the state
        matrix_size = int(self.state[0])  # Get actual matrix size
        sparsity = self.state[1]
        A = generate_dense_matrix(matrix_size) if sparsity > 0.5 else generate_sparse_matrix(matrix_size)
        B = generate_dense_matrix(matrix_size) if sparsity > 0.5 else generate_sparse_matrix(matrix_size)

        # Execute the selected multiplication method
        exec_time, mem_usage = self._execute_multiplication(A, B, method, parallel, partition)
        
        # Reward: Minimize execution time and memory usage
        reward = -exec_time - mem_usage  # Negative reward for optimization goal
      
        # Benchmark: Record the result in the 'results' folder
        if self.save_data:
            benchmark_data = {'method': method, 'exec_time': exec_time, 'mem_usage': mem_usage, 'matrix_size': matrix_size, 'sparsity': sparsity}
            with open(os.path.join(self.results_dir, f'benchmark_{time.time()}.pkl'), 'wb') as f:
                pickle.dump(benchmark_data, f)

        done = False  # Not a terminal task
        return self.state, reward, done, {}

    def _execute_multiplication(self, A, B, method, parallel, partition):
        """Execute matrix multiplication based on the selected method."""
        start_time = time.time()

        if method.lower() == 'naive':
            if parallel:
                result = parallel_multiplication(A, B, method='naive', num_processes=partition)
            else:
                result = naive_multiplication(A, B)
        elif method.lower() == 'strassen':
            if parallel:
                result = parallel_multiplication(A, B, method='strassen', num_processes=partition)
            else:
                result = strassen_multiplication(A, B) 
        elif method.lower() == 'block':
            if parallel:
                result = parallel_multiplication(A, B, method='block', num_processes=partition)
            else:
                result = block_multiplication(A, B)

        exec_time = time.time() - start_time
        mem_usage = A.nbytes + B.nbytes + result.nbytes

        return exec_time, mem_usage
    
    def render(self):
        pass




####################################################################################
if __name__ == "__main__":
    # Initialize environment
    env = MatrixMultiplicationEnv()
    
    # Reset environment
    state = env.reset()
    print(f"Initial State: {state}")

    # Take action and observe results
    action = 0  # Example action
    next_state, reward, done, info = env.step(action)
    print(f"Next State: {next_state}, Reward: {reward}")