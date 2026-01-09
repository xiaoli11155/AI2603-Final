import os
import argparse
import csv
import datetime
import numpy as np
from tqdm import tqdm
import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.policy.policy import Policy
from ChineseChecker import chinese_checker_v0
from agents import GreedyPolicy

def load_policy(checkpoint_path, policy_name='default_policy'):
    # Load the policy from the checkpoint
    policy = Policy.from_checkpoint(checkpoint_path)
    # If it's a policy set, extract the specific policy
    if isinstance(policy, dict):
        policy = policy[policy_name]
    return policy

def evaluate_trials(env, your_policy, baseline_policy, num_matches=30, seed_offset=0, desc="Evaluating"):
    won = 0
    pbar = tqdm(range(num_matches), desc=desc, leave=False)
    for i in pbar:
        seed = seed_offset + i
        env.reset(seed=seed)
        for a in range(len(env.possible_agents)):
            env.action_space(env.possible_agents[a]).seed(seed)
        
        # Game loop
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            else:
                # Choose action based on agent
                if agent == env.possible_agents[0]:
                    action = your_policy.compute_single_action(obs)
                else:
                    action = baseline_policy.compute_single_action(obs)
            
            # Handle different return formats from compute_single_action
            if isinstance(action, (list, tuple, np.ndarray)):
                act = int(action[0])
            else:
                act = int(action)
                
            env.step(act)
            
        # Check winner (assuming player_0 is the first agent)
        if env.unwrapped.winner == env.possible_agents[0]:
            won += 1
        
        pbar.set_postfix({"wins": won, "winrate": f"{won/(i+1):.2%}"})
    
    return won / num_matches

def main():
    parser = argparse.ArgumentParser(description="Test RL Checkpoint against Greedy and RL Baseline over multiple rounds.")
    parser.add_argument('--checkpoint', type=str, default='logs/chinese_checkers_full_sharing0.001_2025-12-20_10-45-41/checkpoint91', help='Path to RL checkpoint to test')
    parser.add_argument('--triangle_size', type=int, default=2, help='Size of the triangle (board scale)')
    parser.add_argument('--num_rounds', type=int, default=100, help='Number of rounds to run')
    parser.add_argument('--matches_per_round', type=int, default=50, help='Number of matches per round per baseline')
    parser.add_argument('--output', type=str, default='checkpoint0_eval_results.csv', help='Output CSV file path')
    args = parser.parse_args()

    # Create environment
    env = chinese_checker_v0.env(render_mode=None, triangle_size=args.triangle_size, max_iters=200)

    # Initialize Greedy Policy
    greedy_policy = GreedyPolicy(args.triangle_size)
    
    # Load RL Baseline
    rl_baseline_path = os.path.join(os.path.dirname(__file__), 'pretrained')
    print(f"Loading RL baseline from {rl_baseline_path}...")
    try:
        rl_baseline_policy = load_policy(rl_baseline_path)
    except Exception as e:
        print(f"Error loading RL baseline: {e}")
        rl_baseline_policy = None

    # Load Your Policy (the checkpoint to test)
    print(f"Loading test checkpoint from {args.checkpoint}...")
    try:
        your_policy = load_policy(args.checkpoint)
    except Exception as e:
        print(f"Error loading test checkpoint: {e}")
        return

    results_greedy = []
    results_rl = []

    print(f"\nEvaluating RL Checkpoint on board size {args.triangle_size}")
    print(f"Protocol: {args.num_rounds} rounds, {args.matches_per_round} matches per round vs each opponent.")

    for r in range(args.num_rounds):
        print(f"\n--- Round {r+1}/{args.num_rounds} ---")
        seed_offset = r * args.matches_per_round + 2000 # Different seed offset
        
        # Test vs Greedy
        wr_greedy = evaluate_trials(
            env, your_policy, greedy_policy, 
            num_matches=args.matches_per_round, 
            seed_offset=seed_offset, 
            desc=f"R{r+1} vs Greedy"
        )
        results_greedy.append(wr_greedy)
        
        # Test vs RL Baseline
        if rl_baseline_policy:
            wr_rl = evaluate_trials(
                env, your_policy, rl_baseline_policy, 
                num_matches=args.matches_per_round, 
                seed_offset=seed_offset + 5000, 
                desc=f"R{r+1} vs RL"
            )
            results_rl.append(wr_rl)
        else:
            results_rl.append(0.0)

    # Final Summary Table
    print("\n" + "="*50)
    print(f"{'ROUND':<10} | {'vs GREEDY':<15} | {'vs RL BASELINE':<15}")
    print("-" * 50)
    for i in range(args.num_rounds):
        rl_str = f"{results_rl[i]:.2%}" if rl_baseline_policy else "N/A"
        print(f"{i+1:<10} | {results_greedy[i]:<15.2%} | {rl_str:<15}")
    
    print("-" * 50)
    avg_greedy = np.mean(results_greedy)
    avg_rl = np.mean(results_rl) if rl_baseline_policy else 0.0
    rl_avg_str = f"{avg_rl:.2%}" if rl_baseline_policy else "N/A"
    print(f"{'AVERAGE':<10} | {avg_greedy:<15.2%} | {rl_avg_str:<15}")
    print("="*50)
    
    # Save to CSV
    print(f"\nSaving results to {args.output}...")
    with open(args.output, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "vs Greedy Winrate", "vs RL Baseline Winrate"])
        for i in range(args.num_rounds):
            rl_val = results_rl[i] if rl_baseline_policy else "N/A"
            writer.writerow([i + 1, results_greedy[i], rl_val])
        writer.writerow(["Average", avg_greedy, avg_rl if rl_baseline_policy else "N/A"])
    
    env.close()

if __name__ == "__main__":
    main()
