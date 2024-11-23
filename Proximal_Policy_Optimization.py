import torch
import torch.nn as nn
import torch.optim as optim

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        
        # Policy network (actor)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)  # Probabilities for each action
        )
        
        # Value network (critic)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Value of the state
        )

    def forward(self, x):
        shared_output = self.shared(x)
        policy = self.actor(shared_output)
        value = self.critic(shared_output)
        return policy, value

# PPO Algorithm
class PPOAgent:
    def __init__(self, env, gamma=0.99, clip_epsilon=0.2, lr=1e-3, epochs=10, batch_size=32):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        self.actor_critic = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def compute_advantage(self, rewards, values, dones):
        """Calculate advantages using Generalized Advantage Estimation (GAE)."""
        advantages = []
        g = 0  # Accumulated return
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            g = r + self.gamma * g * (1 - d)
            advantages.insert(0, g - v)  # Advantage = Gt - Vt
        return advantages

    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        """Update policy and value networks using PPO."""
        for _ in range(self.epochs):
            for idx in range(0, len(states), self.batch_size):
                # Mini-batch sampling
                batch_indices = slice(idx, idx + self.batch_size)
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                policy, values = self.actor_critic(batch_states)
                dist = torch.distributions.Categorical(policy)
                new_log_probs = dist.log_prob(batch_actions)

                # Policy loss with clipping
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratios * batch_advantages, clipped_ratios * batch_advantages).mean()

                # Value loss
                value_loss = ((values - batch_returns) ** 2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, episodes=1000):
        """Train the PPO agent."""
        for episode in range(episodes):
            states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
            state = self.env.reset()
            
            while True:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                policy, value = self.actor_critic(state_tensor)

                # Sample action
                dist = torch.distributions.Categorical(policy)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()

                # Step environment
                next_state, reward, done, _ = self.env.step(action)

                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())
                log_probs.append(log_prob)

                state = next_state

                if done:
                    break
            
            # Compute returns and advantages
            returns = self.compute_advantage(rewards, values, dones)
            advantages = torch.tensor(returns) - torch.tensor(values)
            
            # PPO update
            self.ppo_update(
                torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(log_probs, dtype=torch.float32),
                torch.tensor(returns, dtype=torch.float32),
                torch.tensor(advantages, dtype=torch.float32)
            )

            # Logging
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {sum(rewards)}")


