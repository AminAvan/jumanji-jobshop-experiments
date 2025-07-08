import jax
import jax.numpy as jnp
import jumanji
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import os


class JobShopRunner:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize JobShop environment with specific parameters."""
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'environment': {
                    'name': 'JobShop-v0',
                    'num_jobs': 5,
                    'num_machines': 4,
                    'max_num_ops': 4,
                    'max_op_duration': 4
                }
            }

        # Create JobShop environment with specific parameters
        env_config = self.config['environment']

        # First, let's check what JobShop environments are available
        print("Checking available JobShop environments...")
        envs = jumanji.registered_environments()
        jobshop_envs = [env for env in envs if 'jobshop' in env.lower()]
        print(f"Found JobShop environments: {jobshop_envs}")

        # Try to create the environment with parameters
        try:
            # Try creating with parameters
            self.env = jumanji.make(
                'JobShop-v0',
                num_jobs=env_config['num_jobs'],
                num_machines=env_config['num_machines'],
                max_num_ops=env_config['max_num_ops'],
                max_op_duration=env_config['max_op_duration']
            )
            print(f"Created JobShop environment with custom parameters:")
            print(f"  - num_jobs: {env_config['num_jobs']}")
            print(f"  - num_machines: {env_config['num_machines']}")
            print(f"  - max_num_ops: {env_config['max_num_ops']}")
            print(f"  - max_op_duration: {env_config['max_op_duration']}")
        except Exception as e:
            print(f"Could not create environment with parameters: {e}")
            # Try without parameters
            try:
                self.env = jumanji.make('JobShop-v0')
                print("Created default JobShop environment")
            except:
                if jobshop_envs:
                    self.env = jumanji.make(jobshop_envs[0])
                    print(f"Using environment: {jobshop_envs[0]}")
                else:
                    raise ValueError("No JobShop environment found!")

        # Print environment specifications
        self._print_env_specs()

    def _print_env_specs(self):
        """Print environment specifications."""
        print("\nEnvironment Specifications:")
        print(f"  - Observation spec: {self.env.observation_spec}")
        print(f"  - Action spec: {self.env.action_spec}")

    def run_scenario(self, scenario_seed: int = 0, render: bool = False) -> Dict[str, Any]:
        """Run a single scenario with the specified JobShop configuration."""
        key = jax.random.PRNGKey(scenario_seed)
        state, timestep = self.env.reset(key)

        # Store scenario information
        scenario_info = {
            'seed': scenario_seed,
            'num_jobs': self.config['environment']['num_jobs'],
            'num_machines': self.config['environment']['num_machines'],
            'max_num_ops': self.config['environment']['max_num_ops'],
            'max_op_duration': self.config['environment']['max_op_duration'],
            'steps': [],
            'total_reward': 0,
            'episode_length': 0,
            'makespan': None  # Will be updated at the end
        }

        step_count = 0

        if render:
            self.env.render(state)
            plt.title(f"Initial State - Scenario {scenario_seed}")
            plt.savefig(f"results/scenario_{scenario_seed}_step_0.png")
            plt.show()

        while not timestep.last():
            # Generate random action
            key, action_key = jax.random.split(key)

            # Get valid actions mask if available
            if hasattr(timestep.observation, 'action_mask'):
                action_mask = timestep.observation.action_mask
                # Sample from valid actions only
                valid_actions = jnp.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action_idx = jax.random.choice(action_key, valid_actions)
                    action = int(action_idx)
                else:
                    action = 0  # Default action if no valid actions
            else:
                # Random action from action space
                action = self.env.action_spec.generate_value()

            # Take a step
            state, timestep = self.env.step(state, action)
            step_count += 1

            # Store step information
            step_info = {
                'step': step_count,
                'action': int(action),
                'reward': float(timestep.reward),
                'done': bool(timestep.last())
            }
            scenario_info['steps'].append(step_info)
            scenario_info['total_reward'] += float(timestep.reward)

            if render and step_count <= 10:  # Render first 10 steps
                self.env.render(state)
                plt.title(f"Step {step_count} - Scenario {scenario_seed}")
                plt.savefig(f"results/scenario_{scenario_seed}_step_{step_count}.png")
                plt.show()

        scenario_info['episode_length'] = step_count

        # Extract makespan if available in the final state
        if hasattr(state, 'makespan'):
            scenario_info['makespan'] = float(state.makespan)
        elif hasattr(timestep.observation, 'makespan'):
            scenario_info['makespan'] = float(timestep.observation.makespan)

        return scenario_info

    def run_multiple_scenarios(self, num_scenarios: int = 5, render_first: bool = True) -> List[Dict[str, Any]]:
        """Run multiple scenarios with different random seeds."""
        results = []

        print(f"\nRunning {num_scenarios} scenarios with configuration:")
        print(f"  - num_jobs: {self.config['environment']['num_jobs']}")
        print(f"  - num_machines: {self.config['environment']['num_machines']}")
        print(f"  - max_num_ops: {self.config['environment']['max_num_ops']}")
        print(f"  - max_op_duration: {self.config['environment']['max_op_duration']}")

        for scenario_id in tqdm(range(num_scenarios), desc="Running scenarios"):
            render = render_first and (scenario_id == 0)
            scenario_result = self.run_scenario(scenario_seed=scenario_id, render=render)
            results.append(scenario_result)

        return results

    def analyze_results(self, results: List[Dict[str, Any]]):
        """Analyze and print results from multiple scenarios."""
        print("\n" + "=" * 60)
        print("RESULTS ANALYSIS")
        print("=" * 60)

        # Extract metrics
        rewards = [r['total_reward'] for r in results]
        lengths = [r['episode_length'] for r in results]
        makespans = [r['makespan'] for r in results if r['makespan'] is not None]

        print(f"\nTotal Scenarios Run: {len(results)}")
        print(f"\nEpisode Lengths:")
        print(f"  - Mean: {sum(lengths) / len(lengths):.2f}")
        print(f"  - Min: {min(lengths)}")
        print(f"  - Max: {max(lengths)}")

        print(f"\nTotal Rewards:")
        print(f"  - Mean: {sum(rewards) / len(rewards):.2f}")
        print(f"  - Min: {min(rewards):.2f}")
        print(f"  - Max: {max(rewards):.2f}")

        if makespans:
            print(f"\nMakespans (completion time):")
            print(f"  - Mean: {sum(makespans) / len(makespans):.2f}")
            print(f"  - Min: {min(makespans):.2f}")
            print(f"  - Max: {max(makespans):.2f}")

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Episode lengths
        axes[0].bar(range(len(lengths)), lengths)
        axes[0].set_xlabel('Scenario')
        axes[0].set_ylabel('Episode Length')
        axes[0].set_title('Episode Lengths per Scenario')

        # Rewards
        axes[1].bar(range(len(rewards)), rewards)
        axes[1].set_xlabel('Scenario')
        axes[1].set_ylabel('Total Reward')
        axes[1].set_title('Total Rewards per Scenario')

        plt.tight_layout()
        plt.savefig('results/scenario_analysis.png')
        plt.show()


def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Create runner with configuration
    config_path = 'configs/jobshop_config.yaml'
    runner = JobShopRunner(config_path=config_path)

    # Run multiple scenarios
    results = runner.run_multiple_scenarios(num_scenarios=5, render_first=True)

    # Analyze results
    runner.analyze_results(results)

    # Save detailed results
    import json
    with open('results/scenario_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to 'results/scenario_results.json'")


if __name__ == "__main__":
    main()