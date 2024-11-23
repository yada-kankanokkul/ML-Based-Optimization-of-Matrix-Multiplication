import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from MM_Env import MatrixMultiplicationEnv


# Define DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

def train_dqn(env, num_episodes=500, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, lr=1e-3):
    # Initialize DQN and target DQN
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Optimizer and replay buffer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0

        for t in range(200):  # Max steps per episode
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            # Store transition in replay buffer
            replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done)

            # Move to the next state
            state = next_state
            total_reward += reward

            # Perform a training step
            if len(replay_buffer) > batch_size:
                # Sample a batch of transitions
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Compute Q-values for current states
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute target Q-values
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

                # Compute loss and optimize
                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if t % 10 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return policy_net

if __name__ == "__main__":
    # Initialize the environment
    env = MatrixMultiplicationEnv(matrix_size=[64, 1024],  # Matrix sizes: 64, 1024, 4096
                 multiplication_methods=['naive', 'block'], 
                 parallel_options=[False, True], 
                 partition_options=[2, 4], 
                 save_data=True)

    # Train the DQN
    trained_policy = train_dqn(env, num_episodes=5, lr = 0.1, epsilon_decay= 0.5)

    # Save the trained policy
    torch.save(trained_policy.state_dict(), "dqn_policy.pth")
