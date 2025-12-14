# Timing Benchmarks Summary

## Overview

Created two complementary timing benchmark scripts to measure algorithm performance:

1. **Training Time Benchmark** ([`timing_benchmark.py`](file:///Users/Salman_1/PycharmProjects/dynamicNeighborhoodConstruction/experiments/timing_benchmark.py))
   - Measures **average time per episode** during training
   - Includes both action selection AND model updates (gradients)
   - Use for: Understanding training computational cost

2. **Inference Time Benchmark** ([`inference_timing_benchmark.py`](file:///Users/Salman_1/PycharmProjects/dynamicNeighborhoodConstruction/experiments/inference_timing_benchmark.py))
   - Measures **average time per step** during inference
   - Includes ONLY action selection (no training updates)
   - Use for: Understanding deployment/evaluation performance

---

## Training Time Benchmark

### Purpose
Measure how long each algorithm takes to train per episode (including gradient updates).

### Usage
```bash
# Full benchmark (100 episodes each)
python experiments/timing_benchmark.py \
    --O 75 \
    --n_actions 2 \
    --n_episodes 100 \
    --output_dir ./timing_results

# Quick test (10 episodes)
python experiments/timing_benchmark.py \
    --O 75 \
    --n_actions 2 \
    --n_episodes 10 \
    --output_dir /tmp/timing_test
```

### Algorithms Tested
1. MinMax (baseline)
2. DNC Greedy
3. DNC (SA, cooling=0.1)
4. DNC (SA, cooling=0.5)
5. CMA-ES

### Metrics Reported
- Mean episode time (seconds)
- Median episode time
- Std episode time
- Total training time
- Speedup vs MinMax

### Example Output
```
Algorithm            Mean Time (s)   Median Time (s) Total Time (s) 
-----------------------------------------------------------------
MinMax               0.1234          0.1200          12.34          
DNC Greedy           0.2456          0.2400          24.56          
DNC (SA, cooling=0.1) 0.3678         0.3600          36.78          
DNC (SA, cooling=0.5) 0.3245         0.3150          32.45          
CMA-ES               0.4567          0.4500          45.67          
```

---

## Inference Time Benchmark

### Purpose
Measure how long each algorithm takes per step during inference (action selection only).

### Usage
```bash
# Full benchmark (100 episodes)
python experiments/inference_timing_benchmark.py \
    --O 75 \
    --n_actions 2 \
    --n_episodes 100 \
    --output_dir ./inference_timing_results

# Quick test (10 episodes)
python experiments/inference_timing_benchmark.py \
    --O 75 \
    --n_actions 2 \
    --n_episodes 10 \
    --output_dir /tmp/inference_test
```

### Algorithms Tested
Same 5 algorithms as training benchmark.

### Metrics Reported
- Mean step time (milliseconds)
- Median step time
- 95th percentile (P95)
- 99th percentile (P99)
- Total steps executed
- Speedup/slowdown vs MinMax

### Example Output
```
Algorithm                 Mean (ms)    Median (ms)  P95 (ms)     P99 (ms)    
-----------------------------------------------------------------------------
MinMax                    2.45         2.40         3.20         4.10        
DNC Greedy                4.56         4.50         5.80         6.90        
DNC (SA, cooling=0.1)     8.12         8.00         10.50        12.30       
DNC (SA, cooling=0.5)     7.23         7.10         9.40         11.20       
CMA-ES                    12.34        12.10        15.60        18.40       

Inference speedup compared to MinMax (baseline):
  DNC Greedy                : 1.86x slower (4.56ms vs 2.45ms)
  DNC (SA, cooling=0.1)     : 3.31x slower (8.12ms vs 2.45ms)
  DNC (SA, cooling=0.5)     : 2.95x slower (7.23ms vs 2.45ms)
  CMA-ES                    : 5.04x slower (12.34ms vs 2.45ms)
```

---

## Key Differences

| Aspect | Training Benchmark | Inference Benchmark |
|--------|-------------------|---------------------|
| **Measures** | Time per episode | Time per step |
| **Includes training updates?** | ✅ Yes | ❌ No |
| **Use case** | Estimate training cost | Estimate deployment latency |
| **Typical metric** | Seconds per episode | Milliseconds per step |
| **Training mode** | `training=True` | `training=False` |

---

## When to Use Each

### Use Training Benchmark when:
- Planning computational resources for experiments
- Comparing total training wall-clock time
- Optimizing training hyperparameters
- Deciding feasibility of large-scale experiments

### Use Inference Benchmark when:
- Evaluating real-time deployment feasibility
- Comparing action selection overhead
- Optimizing for production use
- Understanding per-decision latency

---

## Understanding the Results

### Expected Patterns

**Training Time (Episode)**:
- MinMax: Fastest (no neighborhood search)
- DNC Greedy: Moderate (single greedy evaluation)
- DNC SA (higher cooling): Moderate (fewer iterations)
- DNC SA (lower cooling): Slower (more iterations)
- CMA-ES: Slowest (population-based evolution)

**Inference Time (Step)**:
- Same ordering as training
- But: Gradient computation is excluded
- Differences mainly due to neighborhood search complexity

### Cooling Impact
Comparing `cooling=0.1` vs `cooling=0.5`:
- Lower cooling (0.1) → More search iterations → Slower but potentially better quality
- Higher cooling (0.5) → Fewer search iterations → Faster but potentially worse quality

---

## Output Files

Both scripts save JSON results:

**Training**: `timing_benchmark_O{O}_N{n_actions}.json`
**Inference**: `inference_timing_benchmark_O{O}_N{n_actions}.json`

Structure:
```json
[
  {
    "algorithm": "DNC (SA, cooling=0.1)",
    "neighbor_picking": "SA",
    "cooling": 0.1,
    "O": 75,
    "n_actions": 2,
    "n_episodes": 100,
    "mean_step_time": 0.00812,  // or mean_episode_time for training
    "median_step_time": 0.00800,
    "p95_step_time": 0.01050,    // inference only
    "p99_step_time": 0.01230,    // inference only
    ...
  }
]
```