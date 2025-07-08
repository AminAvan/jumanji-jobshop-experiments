import jax
import jax.numpy as jnp
import jumanji
from jumanji.environments.packing.job_shop import JobShop
from jumanji.environments.packing.job_shop.generator import RandomGenerator, ToyGenerator
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import os
import json
import numpy as np


class JobShopRunner:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize JobShop environment with specific parameters."""
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'environment': {
                    'name': 'JobShop-v0',
                    'num_jobs': 5,
                    'num_machines': 4,
                    'max_num_ops': 4,
                    'max_op_duration': 4
                }
            }

        # Create JobShop environment with custom generator
        env_config = self.config['environment']

        # Create a RandomGenerator with your specific parameters
        generator = RandomGenerator(
            num_jobs=env_config['num_jobs'],
            num_machines=env_config['num_machines'],
            max_num_ops=env_config['max_num_ops'],
            max_op_duration=env_config['max_op_duration']
        )

        # Create the JobShop environment with the custom generator
        self.env = JobShop(generator=generator)

        print(f"Created JobShop environment with custom parameters:")
        print(f"  - num_jobs: {self.env.num_jobs}")
        print(f"  - num_machines: {self.env.num_machines}")
        print(f"  - max_num_ops: {self.env.max_num_ops}")
        print(f"  - max_op_duration: {self.env.max_op_duration}")

        # Print environment specifications
        self._print_env_specs()

    def _print_env_specs(self):
        """Print environment specifications."""
        print("\nEnvironment Specifications:")
        print(f"  - Observation spec: {self.env.observation_spec}")
        print(f"\n  - Action spec: {self.env.action_spec}")

    def _select_valid_action(self, key: jax.random.PRNGKey, action_mask: jnp.ndarray) -> jnp.ndarray:
        """Select a valid action based on the action mask.

        The action is an array of shape (num_machines,) where each element is the job ID
        to be scheduled on that machine (or num_jobs for no-op).
        """
        num_machines = action_mask.shape[0]
        actions = []

        for machine_idx in range(num_machines):
            machine_mask = action_mask[machine_idx]
            valid_job_ids = jnp.where(machine_mask)[0]

            if len(valid_job_ids) > 0:
                # Select a random valid job for this machine
                key, subkey = jax.random.split(key)
                action = jax.random.choice(subkey, valid_job_ids)
            else:
                # No valid jobs for this machine, select no-op
                action = self.env.no_op_idx
            actions.append(action)

        return jnp.array(actions, dtype=jnp.int32)

    def run_scenario(self, scenario_seed: int = 0, render: bool = False, max_steps: int = 1000) -> Dict[str, Any]:
        """Run a single scenario."""
        key = jax.random.PRNGKey(scenario_seed)
        state, timestep = self.env.reset(key)

        # Store scenario information
        scenario_info = {
            'seed': scenario_seed,
            'config': self.config['environment'],
            'steps': [],
            'total_reward': 0,
            'episode_length': 0,
            'makespan': None
        }

        step_count = 0

        if render:
            os.makedirs('results/plots', exist_ok=True)
            self.env.render(state)
            plt.savefig(f"results/plots/scenario_{scenario_seed}_step_0.png", dpi=150, bbox_inches='tight')
            plt.close()

        while not timestep.last() and step_count < max_steps:
            # Get action mask from observation
            action_mask = timestep.observation.action_mask

            # Select valid actions for all machines
            key, action_key = jax.random.split(key)
            action = self._select_valid_action(action_key, action_mask)

            # Take a step
            state, timestep = self.env.step(state, action)
            step_count += 1

            # Store step information
            step_info = {
                'step': step_count,
                'action': action.tolist(),
                'reward': float(timestep.reward),
                'done': bool(timestep.last())
            }
            scenario_info['steps'].append(step_info)
            scenario_info['total_reward'] += float(timestep.reward)

            # Debug print for first few steps
            if step_count <= 3:
                print(f"\nStep {step_count}:")
                print(f"  Action: {action}")
                print(f"  Reward: {timestep.reward}")
                print(f"  Done: {timestep.last()}")

            if render and step_count <= 5:  # Render first 5 steps
                self.env.render(state)
                plt.savefig(f"results/plots/scenario_{scenario_seed}_step_{step_count}.png", dpi=150,
                            bbox_inches='tight')
                plt.close()

        scenario_info['episode_length'] = step_count

        # Calculate makespan from the final state
        if hasattr(state, 'scheduled_times'):
            # Find the maximum completion time
            scheduled_times = state.scheduled_times
            ops_durations = state.ops_durations
            ops_mask = state.ops_mask

            completion_times = []
            for job_idx in range(scheduled_times.shape[0]):
                for op_idx in range(scheduled_times.shape[1]):
                    if scheduled_times[job_idx, op_idx] >= 0:
                        start_time = scheduled_times[job_idx, op_idx]
                        duration = ops_durations[job_idx, op_idx]
                        if duration > 0:  # Valid operation
                            completion_time = start_time + duration
                            completion_times.append(float(completion_time))

            if completion_times:
                scenario_info['makespan'] = max(completion_times)
            else:
                scenario_info['makespan'] = 0

        print(f"\nScenario {scenario_seed} completed:")
        print(f"  Steps: {step_count}")
        print(f"  Total reward: {scenario_info['total_reward']:.2f}")
        if scenario_info['makespan']:
            print(f"  Makespan: {scenario_info['makespan']:.2f}")

        return scenario_info

    def run_multiple_scenarios(self, num_scenarios: int = 5, render_first: bool = True) -> List[Dict[str, Any]]:
        """Run multiple scenarios with different random seeds."""
        results = []

        print(f"\nRunning {num_scenarios} scenarios with configuration:")
        print(f"  - num_jobs: {self.env.num_jobs}")
        print(f"  - num_machines: {self.env.num_machines}")
        print(f"  - max_num_ops: {self.env.max_num_ops}")
        print(f"  - max_op_duration: {self.env.max_op_duration}")

        for scenario_id in tqdm(range(num_scenarios), desc="Running scenarios"):
            render = render_first and (scenario_id == 0)
            scenario_result = self.run_scenario(scenario_seed=scenario_id, render=render)
            results.append(scenario_result)

        return results

    def run_with_heuristic(self, scenario_seed: int = 0, heuristic: str = "shortest_first") -> Dict[str, Any]:
        """Run a scenario with a simple heuristic policy."""
        key = jax.random.PRNGKey(scenario_seed)
        state, timestep = self.env.reset(key)

        scenario_info = {
            'seed': scenario_seed,
            'heuristic': heuristic,
            'config': self.config['environment'],
            'steps': [],
            'total_reward': 0,
            'episode_length': 0,
            'makespan': None
        }

        step_count = 0

        while not timestep.last() and step_count < 1000:
            action_mask = timestep.observation.action_mask

            # Apply heuristic
            if heuristic == "shortest_first":
                # For each machine, select the job with shortest operation duration
                action = []
                for machine_idx in range(self.env.num_machines):
                    valid_jobs = jnp.where(action_mask[machine_idx])[0]
                    if len(valid_jobs) > 0:
                        # Get operation durations for valid jobs
                        durations = []
                        for job_id in valid_jobs:
                            if job_id < self.env.num_jobs:  # Not a no-op
                                # Find next operation for this job
                                next_op_idx = jnp.argmax(state.ops_mask[job_id])
                                if state.ops_machine_ids[job_id, next_op_idx] == machine_idx:
                                    durations.append((job_id, state.ops_durations[job_id, next_op_idx]))

                        if durations:
                            # Select job with shortest duration
                            selected_job = min(durations, key=lambda x: x[1])[0]
                        else:
                            # Select no-op if no valid operations
                            selected_job = self.env.no_op_idx
                    else:
                        selected_job = self.env.no_op_idx

                    action.append(selected_job)

                action = jnp.array(action, dtype=jnp.int32)
            else:
                # Random selection (fallback)
                key, action_key = jax.random.split(key)
                action = self._select_valid_action(action_key, action_mask)

            # Take a step
            state, timestep = self.env.step(state, action)
            step_count += 1

            scenario_info['steps'].append({
                'step': step_count,
                'action': action.tolist(),
                'reward': float(timestep.reward),
                'done': bool(timestep.last())
            })
            scenario_info['total_reward'] += float(timestep.reward)

        scenario_info['episode_length'] = step_count

        # Calculate makespan
        if hasattr(state, 'scheduled_times'):
            scheduled_times = state.scheduled_times
            ops_durations = state.ops_durations

            completion_times = []
            for job_idx in range(scheduled_times.shape[0]):
                for op_idx in range(scheduled_times.shape[1]):
                    if scheduled_times[job_idx, op_idx] >= 0:
                        start_time = scheduled_times[job_idx, op_idx]
                        duration = ops_durations[job_idx, op_idx]
                        if duration > 0:
                            completion_times.append(float(start_time + duration))

            if completion_times:
                scenario_info['makespan'] = max(completion_times)

        return scenario_info

    def analyze_results(self, results: List[Dict[str, Any]]):
        """Analyze and print results from multiple scenarios."""
        print("\n" + "=" * 60)
        print("RESULTS ANALYSIS")
        print("=" * 60)

        # Extract metrics
        rewards = [r['total_reward'] for r in results]
        lengths = [r['episode_length'] for r in results]
        makespans = [r.get('makespan', 0) for r in results if r.get('makespan', 0) > 0]

        print(f"\nTotal Scenarios Run: {len(results)}")
        print(f"\nEpisode Lengths:")
        print(f"  - Mean: {np.mean(lengths):.2f}")
        print(f"  - Std: {np.std(lengths):.2f}")
        print(f"  - Min: {min(lengths)}")
        print(f"  - Max: {max(lengths)}")

        print(f"\nTotal Rewards:")
        print(f"  - Mean: {np.mean(rewards):.2f}")
        print(f"  - Std: {np.std(rewards):.2f}")
        print(f"  - Min: {min(rewards):.2f}")
        print(f"  - Max: {max(rewards):.2f}")

        if makespans:
            print(f"\nMakespans (completion time):")
            print(f"  - Mean: {np.mean(makespans):.2f}")
            print(f"  - Std: {np.std(makespans):.2f}")
            print(f"  - Min: {min(makespans):.2f}")
            print(f"  - Max: {max(makespans):.2f}")

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Episode lengths
        axes[0, 0].bar(range(len(lengths)), lengths)
        axes[0, 0].set_xlabel('Scenario')
        axes[0, 0].set_ylabel('Episode Length')
        axes[0, 0].set_title('Episode Lengths per Scenario')
        axes[0, 0].grid(True, alpha=0.3)

        # Rewards
        axes[0, 1].bar(range(len(rewards)), rewards, color='orange')
        axes[0, 1].set_xlabel('Scenario')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].set_title('Total Rewards per Scenario')
        axes[0, 1].grid(True, alpha=0.3)

        # Makespans
        if makespans:
            axes[1, 0].bar(range(len(makespans)), makespans, color='green')
            axes[1, 0].set_xlabel('Scenario')
            axes[1, 0].set_ylabel('Makespan')
            axes[1, 0].set_title('Makespan per Scenario')
            axes[1, 0].grid(True, alpha=0.3)

        # Reward per step
        reward_per_step = [r / l if l > 0 else 0 for r, l in zip(rewards, lengths)]
        axes[1, 1].bar(range(len(reward_per_step)), reward_per_step, color='purple')
        axes[1, 1].set_xlabel('Scenario')
        axes[1, 1].set_ylabel('Reward per Step')
        axes[1, 1].set_title('Average Reward per Step')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/scenario_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\nAnalysis plots saved to 'results/scenario_analysis.png'")


def main():
    # Create results directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # Create runner with configuration
    config_path = 'configs/jobshop_config.yaml'
    runner = JobShopRunner(config_path=config_path)

    print("\n" + "=" * 60)
    print("STARTING JOBSHOP EXPERIMENTS")
    print("=" * 60)

    # Run random policy scenarios
    print("\n1. Running with Random Policy:")
    random_results = runner.run_multiple_scenarios(num_scenarios=5, render_first=True)

    # Run heuristic policy
    print("\n2. Running with Shortest-First Heuristic:")
    heuristic_results = []
    for i in range(5):
        result = runner.run_with_heuristic(scenario_seed=i, heuristic="shortest_first")
        heuristic_results.append(result)
        print(f"  Scenario {i}: Makespan = {result.get('makespan', 0):.0f}, Reward = {result['total_reward']:.0f}")

    # Compare results
    print("\n3. Comparison:")
    random_makespans = [r.get('makespan', 0) for r in random_results if r.get('makespan', 0) > 0]
    heuristic_makespans = [r.get('makespan', 0) for r in heuristic_results if r.get('makespan', 0) > 0]

    if random_makespans and heuristic_makespans:
        print(f"  Random Policy Average Makespan: {np.mean(random_makespans):.2f}")
        print(f"  Heuristic Policy Average Makespan: {np.mean(heuristic_makespans):.2f}")
        improvement = (np.mean(random_makespans) - np.mean(heuristic_makespans)) / np.mean(random_makespans) * 100
        print(f"  Improvement: {improvement:.1f}%")

    # Analyze results
    print("\nRandom Policy Results:")
    runner.analyze_results(random_results)

    # Save detailed results
    all_results = {
        'random_policy': random_results,
        'heuristic_policy': heuristic_results
    }
    with open('results/all_scenario_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nDetailed results saved to 'results/all_scenario_results.json'")
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()