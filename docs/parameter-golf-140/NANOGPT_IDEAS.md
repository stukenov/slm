# Modded-NanoGPT Ideas for Parameter Golf

## HIGH PRIORITY (proven, easy to add)

### 1. Polar Express (replaces Newton-Schulz in Muon)
- Iteration-adaptive coefficients instead of fixed `(3.4445, -4.7750, 2.0315)`
- 5 precomputed coefficient triplets, each step uses its own `(a, b, c)`
- Better orthogonalization quality → better convergence
- Drop-in replacement

### 2. Batch Size Schedule (ramp-up)
- Start small batch (fast steps, noisy gradients), ramp up 2-3x
- Already proven by #236: 524K > 786K in fixed wallclock
- Could go further: start at 262K, ramp to 524K at midpoint

### 3. Multi-Token Prediction (MTP) as auxiliary loss
- Train with next-2 and next-3 token prediction heads
- Anneal auxiliary losses to zero by training end
- Richer training signal at no inference/size cost (heads discarded)
- PR #254 already has MTP infrastructure (mtp_num_heads param)

### 4. Momentum Cooldown
- Ramp momentum back down 0.99→0.85 in last ~50 steps
- Complements momentum warmup already used

## MEDIUM PRIORITY (worth testing)

### 5. NorMuon (per-neuron adaptive LR)
- After Polar Express, normalize each row by second-moment statistics
- beta2=0.95 for per-neuron tracker
- ~11% efficiency improvement over plain Muon

### 6. Attention-Free Middle Layer
- Skip attention in 1 middle layer (use only MLP)
- Saves compute → more training steps in 600s

### 7. Half-Truncated RoPE
- Apply rotary only to first half of head dims
- Second half becomes position-independent (content attention)
- Zero parameter cost

### 8. Residual Scaling (backout lambda)
- Per-sublayer learnable residual scaling, init ~1.1^0.5
- "Backout" mechanism: subtract fraction of early-layer output before prediction
- Negligible parameter cost

## LOW PRIORITY (complex or uncertain)

### 9. Value Embeddings in Attention
- Token-level embeddings mixed into attention values
- Expensive: 512KB per set × 5 sets = 2.5MB → probably too much for 16MB budget

### 10. FP8 MatMul During Training
- Use e4m3fn for lm_head matmul with asymmetric scaling
- Speeds up training → more steps in 600s
- Complex implementation

### 11. Sliding Window Attention in Training (not just eval)
- Progressive widening: 128→1664 tokens during training
- With YaRN for RoPE extension
- Complex schedule management
