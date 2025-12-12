"""
Baseline Solver: Optimal (s,S) Policy for Independent Items
Solves for optimal s and S parameters for each item independently
"""
import numpy as np
from scipy.optimize import minimize
from Environments.InventoryControl.JointReplenishment_py import JointReplenishment_py


def solve_optimal_sS_single_item(K, h, b, lambda_demand, smin=0, smax=66, max_steps=100, replications=50):
    """
    Solve optimal (s,S) for a single item using grid search
    
    Args:
        K: Fixed order cost (common order cost)
        h: Holding cost per unit
        b: Backorder cost per unit  
        lambda_demand: Poisson demand parameter
        smin, smax: Order quantity bounds
        max_steps: Episode length
        replications: Number of simulation runs for evaluation
    
    Returns:
        (s_opt, S_opt, cost_opt): Optimal parameters and cost
    """
    best_cost = float('inf')
    best_s, best_S = None, None
    
    # Create environment once (more efficient)
    env = JointReplenishment_py(
        smin=smin, smax=smax, n_items=1, 
        commonOrderCosts=K, max_steps=max_steps,
        mappingType='dnc_mapping'
    )
    
    # Set costs for this item
    env.h[0] = h
    env.b[0] = b
    env.d_lambda[0] = lambda_demand
    
    # Grid search over (s, S) pairs
    print(f"  Searching optimal (s,S) for K={K}, h={h}, b={b}, λ={lambda_demand}...")
    total_evals = (smax - smin + 1) * (smax - smin + 2) // 2
    eval_count = 0
    
    for s in range(smin, smax + 1):
        for S in range(s, smax + 1):
            # Evaluate policy
            cost = -env.simPolicy([s], [S], replications=replications)  # Negative because rewards are costs
            
            if cost < best_cost:
                best_cost = cost
                best_s, best_S = s, S
            
            eval_count += 1
            if eval_count % 100 == 0:
                print(f"    Progress: {eval_count}/{total_evals} evaluations...")
    
    print(f"  Found optimal: s={best_s}, S={best_S}, cost={best_cost:.2f}")
    return best_s, best_S, best_cost


def solve_independent_sS(n_items, K, h_even, b_even, h_uneven, b_uneven, 
                         lambda_even, lambda_uneven, smin=0, smax=66, 
                         max_steps=100, replications=50):
    """
    Solve optimal (s,S) for each item independently
    
    Returns:
        s_list: List of optimal s values [s0, s1, ...]
        S_list: List of optimal S values [S0, S1, ...]
        total_cost: Total expected cost
    """
    s_list = []
    S_list = []
    total_cost = 0
    
    for i in range(n_items):
        if i % 2 == 0:
            h, b, lam = h_even, b_even, lambda_even
        else:
            h, b, lam = h_uneven, b_uneven, lambda_uneven
        
        s, S, cost = solve_optimal_sS_single_item(
            K=K, h=h, b=b, lambda_demand=lam,
            smin=smin, smax=smax, max_steps=max_steps,
            replications=replications
        )
        
        s_list.append(s)
        S_list.append(S)
        total_cost += cost
    
    return s_list, S_list, total_cost


def evaluate_baseline_policy(n_items, K, s_list, S_list, max_steps=100, 
                             n_eval_episodes=100):
    """
    Evaluate the baseline (s,S) policy
    
    Returns:
        mean_cost: Average total cost
        std_cost: Standard deviation of costs
        cost_list: List of episode costs
    """
    env = JointReplenishment_py(
        smin=0, smax=66, n_items=n_items,
        commonOrderCosts=K, max_steps=max_steps,
        mappingType='dnc_mapping'
    )
    
    cost_list = []
    
    for episode in range(n_eval_episodes):
        total_cost = 0
        state = env.reset()
        done = False
        
        while not done:
            # Apply (s,S) policy
            action = []
            for i in range(n_items):
                if env.cur_inv[i] <= s_list[i % len(s_list)]:
                    action.append(S_list[i % len(S_list)])
                else:
                    action.append(0)
            
            next_state, reward, done, info = env.step(action=action)
            total_cost -= reward  # Convert reward (negative) to cost (positive)
        
        cost_list.append(total_cost)
    
    return np.mean(cost_list), np.std(cost_list), cost_list


if __name__ == "__main__":
    # Test the solver
    n_items = 2
    K = 75
    s_list, S_list, cost = solve_independent_sS(
        n_items=n_items, K=K,
        h_even=-1, b_even=-19, h_uneven=-1, b_uneven=-19,
        lambda_even=20, lambda_uneven=10,
        replications=20
    )
    
    print(f"Optimal s: {s_list}")
    print(f"Optimal S: {S_list}")
    print(f"Estimated cost: {cost}")

