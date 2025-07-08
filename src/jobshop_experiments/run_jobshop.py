import jax
import jax.numpy as jnp
import jumanji
from jumanji.environments.packing.job_shop import JobShop
from jumanji.environments.packing.job_shop.generator import RandomGenerator, ToyGenerator
# Fix the import - import directly from the actor_critic module
from jumanji.training.networks.job_shop.actor_critic import make_actor_critic_networks_job_shop
from jumanji.training.networks.actor_critic import ActorCriticNetworks
import haiku as hk
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import os
import json
import numpy as np
from dataclasses import dataclass
import optax
from functools import partial
import chex


@dataclass
class PPOConfig:
    """Configuration for PPO training matching the multi-agent scenario."""
    # Environment config
    num_jobs: int = 5
    num_machines: int = 4
    max_num_ops: int = 4
    max_op_duration: int = 4

    # Network architecture (matching multi-agent config)
    num_layers_machines: int = 1
    num_layers_operations: int = 1
    num_layers_joint_machines_jobs: int = 2
    transformer_num_heads: int = 8
    transformer_key_size: int = 16
    transformer_mlp_units: List[int] = None

    # Training config
    num_epochs: int = 65
    num_learner_steps_per_epoch: int = 200
    n_steps: int = 10
    total_batch_size: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4

    # Evaluation config
    eval_episodes: int = 32
    eval_frequency: int = 5

    # PPO hyperparameters
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    def __post_init__(self):
        if self.transformer_mlp_units is None:
            self.transformer_mlp_units = [512]


class PPOTrajectory(NamedTuple):
    """Store trajectory data for PPO training."""
    observations: Any  # JobShop observations
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    log_probs: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray


class PPOJobShopRL:
    """PPO implementation using Jumanji's official JobShop networks."""

    def __init__(self, config: PPOConfig):
        self.config = config

        # Create environment
        generator = RandomGenerator(
            num_jobs=config.num_jobs,
            num_machines=config.num_machines,
            max_num_ops=config.max_num_ops,
            max_op_duration=config.max_op_duration
        )
        self.env = JobShop(generator=generator)

        # Create official Jumanji networks
        self.networks = make_actor_critic_networks_job_shop(
            job_shop=self.env,
            num_layers_machines=config.num_layers_machines,
            num_layers_operations=config.num_layers_operations,
            num_layers_joint_machines_jobs=config.num_layers_joint_machines_jobs,
            transformer_num_heads=config.transformer_num_heads,
            transformer_key_size=config.transformer_key_size,
            transformer_mlp_units=config.transformer_mlp_units,
        )

        # Initialize optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate, eps=1e-5)
        )

        print(f"PPO with Official JobShop Networks:")
        print(f"  - Environment: {config.num_jobs} jobs, {config.num_machines} machines")
        print(f"  - Network: Transformer-based with {config.transformer_num_heads} heads")
        print(f"  - Training epochs: {config.num_epochs}")
        print(f"  - PPO clip epsilon: {config.clip_epsilon}")

    def init_params(self, key: chex.PRNGKey):
        """Initialize network parameters."""
        key1, key2 = jax.random.split(key)

        # Get a dummy observation to initialize networks
        dummy_state, dummy_timestep = self.env.reset(key1)
        dummy_obs = dummy_timestep.observation

        # Initialize both actor and critic networks
        policy_params = self.networks.policy_network.init(key2, dummy_obs)
        value_params = self.networks.value_network.init(key2, dummy_obs)

        return {'policy': policy_params, 'value': value_params}

    @partial(jax.jit, static_argnums=(0,))
    def act(self, params, obs, key):
        """Select action using the policy network."""
        # Get action logits from policy network
        logits = self.networks.policy_network.apply(params['policy'], obs)

        # Sample action from the distribution
        action_dist = self.networks.parametric_action_distribution.create_dist(logits)
        action = action_dist.sample(seed=key)
        log_prob = action_dist.log_prob(action)

        # Get value from critic network
        value = self.networks.value_network.apply(params['value'], obs)

        # Compute entropy for regularization
        entropy = action_dist.entropy()

        return action, log_prob, value, entropy

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_actions(self, params, obs, actions):
        """Evaluate actions for PPO loss computation."""
        # Get action logits and values
        logits = self.networks.policy_network.apply(params['policy'], obs)
        values = self.networks.value_network.apply(params['value'], obs)

        # Create distribution and compute log probs
        action_dist = self.networks.parametric_action_distribution.create_dist(logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return log_probs, values, entropy

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = jnp.zeros(T)
        lastgaelam = 0

        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones[-1]
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.config.discount_factor * nextvalues * nextnonterminal - values[t]
            advantages = advantages.at[t].set(
                delta + self.config.discount_factor * self.config.gae_lambda * nextnonterminal * lastgaelam
            )
            lastgaelam = advantages[t]

        returns = advantages + values
        return advantages, returns

    def collect_trajectories(self, params, key, num_envs):
        """Collect trajectories using current policy."""
        # Initialize environments
        keys = jax.random.split(key, num_envs + 1)
        key = keys[0]

        # Reset all environments
        reset_keys = keys[1:]
        states = []
        timesteps = []

        for i in range(num_envs):
            state, timestep = self.env.reset(reset_keys[i])
            states.append(state)
            timesteps.append(timestep)

        # Storage
        all_observations = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []

        # Collect n_steps of experience
        for step in range(self.config.n_steps):
            # Store observations
            observations = [ts.observation for ts in timesteps]
            all_observations.append(observations)

            # Get actions for all environments
            actions = []
            log_probs = []
            values = []
            entropies = []

            keys = jax.random.split(key, num_envs + 1)
            key = keys[0]

            for i in range(num_envs):
                action, log_prob, value, entropy = self.act(
                    params, timesteps[i].observation, keys[i + 1]
                )
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)

            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)

            # Step environments
            rewards = []
            dones = []

            for i in range(num_envs):
                state, timestep = self.env.step(states[i], actions[i])
                states[i] = state
                timesteps[i] = timestep
                rewards.append(timestep.reward)
                dones.append(timestep.last())

            all_rewards.append(rewards)
            all_dones.append(dones)

        # Convert lists to arrays and compute advantages
        trajectories_data = {
            'observations': all_observations,
            'actions': jnp.array(all_actions),
            'rewards': jnp.array(all_rewards),
            'dones': jnp.array(all_dones),
            'values': jnp.array(all_values),
            'log_probs': jnp.array(all_log_probs)
        }

        # Compute advantages for each environment
        advantages_list = []
        returns_list = []

        for env_idx in range(num_envs):
            env_rewards = trajectories_data['rewards'][:, env_idx]
            env_values = trajectories_data['values'][:, env_idx]
            env_dones = trajectories_data['dones'][:, env_idx]

            adv, ret = self.compute_gae(env_rewards, env_values, env_dones)
            advantages_list.append(adv)
            returns_list.append(ret)

        advantages = jnp.stack(advantages_list).T
        returns = jnp.stack(returns_list).T

        # Normalize advantages
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return PPOTrajectory(
            observations=trajectories_data['observations'],
            actions=trajectories_data['actions'],
            rewards=trajectories_data['rewards'],
            dones=trajectories_data['dones'],
            values=trajectories_data['values'],
            log_probs=trajectories_data['log_probs'],
            advantages=advantages,
            returns=returns
        )

    def ppo_loss(self, params, observations, actions, old_log_probs, advantages, returns):
        """Compute PPO loss for a batch."""
        # Evaluate actions with current policy
        new_log_probs, values, entropy = jax.vmap(
            partial(self.evaluate_actions, params)
        )(observations, actions)

        # PPO clipped objective
        ratio = jnp.exp(new_log_probs - old_log_probs)
        clipped_ratio = jnp.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
        policy_loss = -jnp.minimum(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        # Value loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
                policy_loss +
                self.config.value_loss_coef * value_loss +
                self.config.entropy_coef * entropy_loss
        )

        metrics = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': -entropy_loss,
            'total_loss': total_loss,
            'clip_fraction': (jnp.abs(ratio - 1) > self.config.clip_epsilon).mean()
        }

        return total_loss, metrics

    def update_ppo(self, params, opt_state, trajectories):
        """Perform PPO update on collected trajectories."""
        # Flatten trajectories for batch processing
        n_steps, n_envs = trajectories.actions.shape[:2]
        batch_size = n_steps * n_envs

        # Prepare batch data
        batch_observations = []
        for step in range(n_steps):
            for env in range(n_envs):
                batch_observations.append(trajectories.observations[step][env])

        batch_actions = trajectories.actions.reshape(batch_size, -1)
        batch_log_probs = trajectories.log_probs.reshape(batch_size)
        batch_advantages = trajectories.advantages.reshape(batch_size)
        batch_returns = trajectories.returns.reshape(batch_size)

        # Multiple epochs of updates
        for epoch in range(self.config.update_epochs):
            # Shuffle data
            key = jax.random.PRNGKey(epoch)
            indices = jax.random.permutation(key, batch_size)

            # Update in minibatches
            mb_size = batch_size // self.config.num_minibatches

            for start in range(0, batch_size, mb_size):
                end = min(start + mb_size, batch_size)
                mb_indices = indices[start:end]

                # Get minibatch
                mb_observations = [batch_observations[i] for i in mb_indices]
                mb_actions = batch_actions[mb_indices]
                mb_log_probs = batch_log_probs[mb_indices]
                mb_advantages = batch_advantages[mb_indices]
                mb_returns = batch_returns[mb_indices]

                # Compute loss and update
                (loss, metrics), grads = jax.value_and_grad(
                    self.ppo_loss, has_aux=True
                )(params, mb_observations, mb_actions, mb_log_probs, mb_advantages, mb_returns)

                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

        return params, opt_state, metrics

    def evaluate(self, params, num_episodes=32):
        """Evaluate the current policy."""
        total_rewards = []
        makespans = []
        episode_lengths = []

        for i in range(num_episodes):
            key = jax.random.PRNGKey(i * 1000)
            state, timestep = self.env.reset(key)

            episode_reward = 0
            steps = 0

            while not timestep.last() and steps < 1000:
                key, action_key = jax.random.split(key)
                action, _, _, _ = self.act(params, timestep.observation, action_key)
                state, timestep = self.env.step(state, action)
                episode_reward += float(timestep.reward)
                steps += 1

            total_rewards.append(episode_reward)
            episode_lengths.append(steps)

            # Calculate makespan
            if hasattr(state, 'scheduled_times') and hasattr(state, 'ops_durations'):
                completion_times = []
                for job_idx in range(state.scheduled_times.shape[0]):
                    for op_idx in range(state.scheduled_times.shape[1]):
                        if state.scheduled_times[job_idx, op_idx] >= 0:
                            start_time = state.scheduled_times[job_idx, op_idx]
                            duration = state.ops_durations[job_idx, op_idx]
                            if duration > 0:
                                completion_times.append(float(start_time + duration))

                if completion_times:
                    makespans.append(max(completion_times))

        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_makespan': np.mean(makespans) if makespans else 0,
            'std_makespan': np.std(makespans) if makespans else 0,
            'mean_episode_length': np.mean(episode_lengths)
        }

    def train(self):
        """Main PPO training loop."""
        # Initialize
        key = jax.random.PRNGKey(42)
        key, init_key = jax.random.split(key)
        params = self.init_params(init_key)
        opt_state = self.optimizer.init(params)

        # Training history
        history = {
            'epoch': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clip_fraction': [],
            'mean_reward': [],
            'mean_makespan': []
        }

        # Initial evaluation
        print("\nInitial Policy Evaluation:")
        init_results = self.evaluate(params, num_episodes=32)
        print(f"  Mean Reward: {init_results['mean_reward']:.2f}")
        print(f"  Mean Makespan: {init_results['mean_makespan']:.2f}")

        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_metrics = {
                'policy_loss': [],
                'value_loss': [],
                'entropy': [],
                'clip_fraction': []
            }

            # Multiple steps per epoch
            for step in tqdm(range(self.config.num_learner_steps_per_epoch),
                             desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"):

                # Collect trajectories
                key, collect_key = jax.random.split(key)
                num_envs = self.config.total_batch_size // self.config.n_steps
                trajectories = self.collect_trajectories(params, collect_key, num_envs)

                # PPO update
                params, opt_state, metrics = self.update_ppo(params, opt_state, trajectories)

                # Store metrics
                for k, v in metrics.items():
                    if k != 'total_loss':
                        epoch_metrics[k].append(float(v))

            # Evaluate periodically
            if (epoch + 1) % self.config.eval_frequency == 0:
                eval_results = self.evaluate(params, num_episodes=self.config.eval_episodes)

                # Store history
                history['epoch'].append(epoch + 1)
                history['mean_reward'].append(eval_results['mean_reward'])
                history['mean_makespan'].append(eval_results['mean_makespan'])

                for k, v in epoch_metrics.items():
                    history[k].append(np.mean(v))

                # Print progress
                print(f"\nEpoch {epoch + 1}:")
                print(f"  Policy Loss: {np.mean(epoch_metrics['policy_loss']):.4f}")
                print(f"  Value Loss: {np.mean(epoch_metrics['value_loss']):.4f}")
                print(f"  Entropy: {np.mean(epoch_metrics['entropy']):.4f}")
                print(f"  Clip Fraction: {np.mean(epoch_metrics['clip_fraction']):.4f}")
                print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                print(f"  Mean Makespan: {eval_results['mean_makespan']:.2f} ± {eval_results['std_makespan']:.2f}")

        # Final evaluation
        print("\nFinal Policy Evaluation:")
        final_results = self.evaluate(params, num_episodes=100)
        print(f"  Mean Reward: {final_results['mean_reward']:.2f} ± {final_results['std_reward']:.2f}")
        print(f"  Mean Makespan: {final_results['mean_makespan']:.2f} ± {final_results['std_makespan']:.2f}")

        return params, history, init_results, final_results

    def visualize_training(self, history):
        """Visualize PPO training progress."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        epochs = history['epoch']

        # Policy loss
        axes[0, 0].plot(epochs, history['policy_loss'])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Policy Loss')
        axes[0, 0].set_title('PPO Policy Loss')
        axes[0, 0].grid(True)

        # Value loss
        axes[0, 1].plot(epochs, history['value_loss'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Value Loss')
        axes[0, 1].set_title('Value Function Loss')
        axes[0, 1].grid(True)

        # Entropy
        axes[0, 2].plot(epochs, history['entropy'])
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Entropy')
        axes[0, 2].set_title('Policy Entropy')
        axes[0, 2].grid(True)

        # Clip fraction
        axes[1, 0].plot(epochs, history['clip_fraction'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Clip Fraction')
        axes[1, 0].set_title('PPO Clip Fraction')
        axes[1, 0].grid(True)

        # Reward
        axes[1, 1].plot(epochs, history['mean_reward'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Mean Reward')
        axes[1, 1].set_title('Average Episode Reward')
        axes[1, 1].grid(True)

        # Makespan
        axes[1, 2].plot(epochs, history['mean_makespan'])
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Mean Makespan')
        axes[1, 2].set_title('Average Makespan')
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig('results/ppo_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\nTraining curves saved to 'results/ppo_training_curves.png'")


def main():
    """Run PPO with official Jumanji networks."""
    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Create PPO configuration
    config = PPOConfig()

    print("=" * 60)
    print("PPO WITH OFFICIAL JUMANJI NETWORKS")
    print("Single-Agent JobShop RL")
    print("=" * 60)

    # Create and train PPO agent
    agent = PPOJobShopRL(config)

    # Train
    print("\nStarting PPO Training with Transformer Networks...")
    trained_params, history, init_results, final_results = agent.train()

    # Visualize results
    agent.visualize_training(history)

    # Calculate improvements
    reward_improvement = final_results['mean_reward'] - init_results['mean_reward']
    makespan_improvement = init_results['mean_makespan'] - final_results['mean_makespan']

    # Save comprehensive results
    results = {
        'algorithm': 'PPO with Official Jumanji Networks',
        'network_architecture': {
            'type': 'Transformer-based',
            'num_layers_machines': config.num_layers_machines,
            'num_layers_operations': config.num_layers_operations,
            'num_layers_joint_machines_jobs': config.num_layers_joint_machines_jobs,
            'transformer_num_heads': config.transformer_num_heads,
            'transformer_key_size': config.transformer_key_size,
            'transformer_mlp_units': config.transformer_mlp_units
        },
        'config': {
            'num_jobs': config.num_jobs,
            'num_machines': config.num_machines,
            'max_num_ops': config.max_num_ops,
            'max_op_duration': config.max_op_duration,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'clip_epsilon': config.clip_epsilon,
            'discount_factor': config.discount_factor
        },
        'initial_performance': init_results,
        'final_performance': final_results,
        'training_history': history,
        'improvements': {
            'reward': reward_improvement,
            'makespan': makespan_improvement,
            'reward_percentage': (reward_improvement / abs(init_results['mean_reward'])) * 100 if init_results[
                                                                                                      'mean_reward'] != 0 else 0,
            'makespan_percentage': (makespan_improvement / init_results['mean_makespan']) * 100 if init_results[
                                                                                                       'mean_makespan'] != 0 else 0
        }
    }

    with open('results/ppo_official_networks_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to 'results/ppo_official_networks_results.json'")
    print(f"\nTraining Summary:")
    print(f"  Reward Improvement: {reward_improvement:.2f} ({results['improvements']['reward_percentage']:.1f}%)")
    print(f"  Makespan Improvement: {makespan_improvement:.2f} ({results['improvements']['makespan_percentage']:.1f}%)")

    print("\n" + "=" * 60)
    print("NETWORK ARCHITECTURE INSIGHTS")
    print("=" * 60)
    print("\nOfficial Jumanji Networks use:")
    print("✓ Transformer blocks for machine embeddings")
    print("✓ Self-attention between operations")
    print("✓ Cross-attention between operations and machines")
    print("✓ Positional encoding for operation sequences")
    print("✓ Separate embeddings for jobs and machines")
    print("✓ Multi-head attention mechanisms")
    print("\nThis sophisticated architecture is designed to:")
    print("- Capture complex dependencies between operations")
    print("- Respect the sequential nature of job operations")
    print("- Model machine-operation compatibility")
    print("- Handle variable-length operation sequences")


if __name__ == "__main__":
    main()