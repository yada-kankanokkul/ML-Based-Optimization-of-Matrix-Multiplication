import gym
from gym import spaces
import numpy as np

from Matrix_generation import generate_dense_matrix, generate_sparse_matrix
from Matrix_Multiplication import naive_multiplication, block_multiplication, strassen_multiplication
from Parallel_Multiplication import parallel_multiplication
from Reinforcement_Learning import calculate_reward
from data_collection import benchmark_matrix_multiplication
#benchmark_matrix_multiplication


class MatrixMultiplicationEnv(gym.Env):
    def __init__(self):
        super(MatrixMultiplicationEnv, self).__init__()
        # Action Space: 4 actions (e.g., Naive, Strassen, Block, Parallel)
        self.action_space = spaces.Discrete(4)
        # Observation space: matrix size, sparsity, and type (dense/sparse)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.current_step = 0

    def reset(self):
        # Reset the environment to a random state
        self.state = [100, 0.1, 1]  # Example state: 100x100, 10% sparsity, no partitioning
        return np.array(self.state)
    
    # def reset(self):
    #     # Random state: [matrix size (normalized), sparsity, partitioning]
    #     self.state = np.array([
    #         np.random.choice([100, 500, 1000, 5000]) / 5000,  # Random matrix size normalized to [0.02, 1.0]
    #         np.random.uniform(0.1, 0.9),                     # Random sparsity between 10% and 90%
    #         np.random.choice([0, 1])                        # Random partitioning: 0 (no partition) or 1 (partitioned)
    #     ])
    #     return self.state

    
    def step(self, action):
        # Get matrix based on state properties (size, sparsity)
        size = int(self.state[0])
        sparsity = self.state[1]
        A = generate_sparse_matrix(size, sparsity)
        B = generate_sparse_matrix(size, sparsity)
        
        # Perform the action (choose multiplication method)
        if action == 0:  # Naive multiplication
            result = naive_multiplication(A, B)
        elif action == 1:  # Strassen's algorithm
            result = strassen_multiplication(A, B)
        elif action == 2:  # Block multiplication
            result = block_multiplication(A, B)
        elif action == 3:  # Parallelization
            result = parallel_multiplication(A, B)
        
        # Measure performance
        exec_time, mem_usage, _ = benchmark_matrix_multiplication(A, B, result)
        
        # Calculate the reward
        reward = calculate_reward(exec_time, mem_usage)
        
        
        # New state (for simplicity, keep it the same here)
        done = False  # Not a terminal task
        # Return the new state, reward, and done flag
        return np.array(self.state), reward, done, {}
    
    def render(self):
        pass


####################################################################################
