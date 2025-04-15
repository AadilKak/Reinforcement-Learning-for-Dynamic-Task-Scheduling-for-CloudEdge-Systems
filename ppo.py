import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

class PPO:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.critic_discount = 0.5
        self.entropy_beta = 0.01
        self.actor_lr = 0.0003
        self.critic_lr = 0.001
        self.update_epochs = 10
        self.batch_size = 64
        
        # Build actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Optimizers
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)
        
    def _build_actor(self):
        """Build policy network (actor)"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        
        # Output layer with softmax to get action probabilities
        action_probs = Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=action_probs)
        return model
    
    def _build_critic(self):
        """Build value network (critic)"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        
        # Value prediction
        value = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=value)
        return model
    
    def get_action(self, state):
        """Select action according to the policy"""
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        # Sample action from the probability distribution
        action = np.random.choice(self.action_dim, p=action_probs)
        
        return action, action_probs
    
    def get_value(self, state):
        """Get value estimate from critic"""
        state = np.reshape(state, [1, self.state_dim])
        return self.critic.predict(state, verbose=0)[0, 0]
    
    @tf.function
    def _actor_loss(self, states, actions, advantages, old_probs):
        """Compute PPO actor loss with clipping"""
        actions_one_hot = tf.one_hot(actions, self.action_dim)
        
        # Current policy log probabilities
        new_probs = self.actor(states, training=True)
        new_log_probs = tf.reduce_sum(tf.math.log(new_probs + 1e-10) * actions_one_hot, axis=1)
        old_log_probs = tf.math.log(old_probs + 1e-10)
        
        # Ratio of new and old policies
        ratio = tf.exp(new_log_probs - old_log_probs)
        
        # Clipped objective
        clip_adv = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        loss_clip = tf.minimum(ratio * advantages, clip_adv)
        
        # Add entropy bonus for exploration
        entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1)
        
        return -tf.reduce_mean(loss_clip + self.entropy_beta * entropy)
    
    @tf.function
    def _critic_loss(self, states, returns):
        """Compute critic loss"""
        value_pred = self.critic(states, training=True)
        mse = tf.square(returns - tf.squeeze(value_pred))
        return tf.reduce_mean(mse)
    
    def train(self, states, actions, rewards, next_states, dones, action_probs):
        """Train actor and critic networks using PPO algorithm"""
        # Convert to tensors
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        old_action_probs = np.array([probs[act] for probs, act in zip(action_probs, actions)])
        
        # Calculate advantages and returns
        values = [self.get_value(state) for state in states]
        next_values = [self.get_value(next_state) for next_state in next_states]
        
        # GAE calculation
        advantages = []
        returns = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                delta = rewards[i] + self.gamma * next_values[i] - values[i]
                gae = delta + self.gamma * self.lam * gae
            
            advantages.append(gae)
            returns.append(gae + values[i])
        
        advantages = np.array(advantages[::-1])
        returns = np.array(returns[::-1])
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Training with minibatches for multiple epochs
        indices = np.arange(len(states))
        
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                if end > len(states):
                    end = len(states)
                
                batch_indices = indices[start:end]
                
                with tf.GradientTape() as actor_tape:
                    actor_loss = self._actor_loss(
                        tf.convert_to_tensor(states[batch_indices], dtype=tf.float32),
                        tf.convert_to_tensor(actions[batch_indices], dtype=tf.int32),
                        tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32),
                        tf.convert_to_tensor(old_action_probs[batch_indices], dtype=tf.float32)
                    )
                
                with tf.GradientTape() as critic_tape:
                    critic_loss = self._critic_loss(
                        tf.convert_to_tensor(states[batch_indices], dtype=tf.float32),
                        tf.convert_to_tensor(returns[batch_indices], dtype=tf.float32)
                    )
                
                # Compute and apply gradients
                actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
                
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return actor_loss, critic_loss

# Function to train the PPO agent
def train_ppo():
    env = CloudEdgeEnvironment()
    state_dim = len(env.reset())
    action_dim = 2  # Process on edge or offload to cloud
    
    agent = PPOAgent(state_dim, action_dim)
    episodes = 1000
    max_steps = 100
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        # Storage for experience
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        action_probs = []
        
        for step in range(max_steps):
            # Get action and its probability
            action, probs = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            action_probs.append(probs)
            
            episode_reward += reward
            state = next_state
            
            if len(states) >= 256:  # Batch size for updating
                actor_loss, critic_loss = agent.train(states, actions, rewards, next_states, dones, action_probs)
                
                # Clear storage
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []
                action_probs = []
        
        # Update with remaining data if any
        if states:
            agent.train(states, actions, rewards, next_states, dones, action_probs)
        
        # Log progress
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {episode_reward}")
    
    return agent

if __name__ == "__main__":
    trained_agent = train_ppo()