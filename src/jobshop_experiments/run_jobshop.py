import jax
import jax.numpy as jnp
import jumanji
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm


class JobShopRunner:
    def __init__(self, env_name: str = "JobShop-v0"):
        """Initialize JobShop environment."""
        # First, let's check available environments
        print("Available Jumanji environments:")
        envs = jumanji.registered_environments()
        jobshop_envs = [env for env in envs if 'jobshop' in env.lower()]
        print(f"JobShop environments: {jobshop_envs}")

        # Try to create the environment
        try:
            self.env = jumanji.make(env_name)
        except:
            # If specific name doesn't work, try a general one
            if jobshop_envs:
                self.env = jumanji.make(jobshop_envs[0])
                print(f"Using environment: {jobshop_envs[0]}")
            else:
                raise ValueError("No JobShop environment found!")

    def run_random_policy(self, num_episodes: int = 10, seed: int = 0) -> List[Dict[str, Any]]:
        """Run episodes with random policy."""
        results = []
        key = jax.random.PRNGKey(seed)

        for episode in tqdm(range(num_episodes), desc="Running episodes"):
            key, reset_key = jax.random.split(key)
            state, timestep = self.env.reset(reset_key)

            episode_reward = 0
            episode_length = 0

            while not timestep.last():
                # Random action selection
                key, action_key = jax.random.split(key)
                action = self.env.action_spec.generate_value()

                # Take a step
                state, timestep = self.env.step(state, action)

                episode_reward += timestep.reward
                episode_length += 1

            results.append({
                'episode': episode,
                'total_reward': float(episode_reward),
                'episode_length': episode_length
            })

        return results

    def visualize_episode(self, seed: int = 0):
        """Visualize one episode."""
        key = jax.random.PRNGKey(seed)
        state, timestep = self.env.reset(key)

        # Render initial state
        self.env.render(state)
        plt.title("Initial JobShop State")
        plt.show()

        # Run a few steps
        for step in range(5):
            if timestep.last():
                break

            action = self.env.action_spec.generate_value()
            state, timestep = self.env.step(state, action)

            self.env.render(state)
            plt.title(f"Step {step + 1}")
            plt.show()


def main():
    # Create runner
    runner = JobShopRunner()

    # Run random policy experiments
    results = runner.run_random_policy(num_episodes=5)

    # Print results
    print("\nResults:")
    for result in results:
        print(f"Episode {result['episode']}: "
              f"Reward = {result['total_reward']:.2f}, "
              f"Length = {result['episode_length']}")

    # Visualize one episode
    print("\nVisualizing an episode...")
    runner.visualize_episode()


if __name__ == "__main__":
    main()