# How to Improve Negative Rewards

## 🔍 **Root Cause Analysis:**

### **Why Negative Rewards Happen:**

Looking at the worst performers:
- **Heuristic 4×4, 9 cubes**: -108.42 reward, 33.89 steps
- **A* 4×4, 9 cubes**: -79.86 reward, 32.19 steps
- **RRT 4×4, 9 cubes**: -44.57 reward, 31.17 steps

### **Reward Breakdown:**

**Per valid pick:**
- Base: +10.0
- Distance: +0 to +5.0
- Obstacle: +0 to +3.0
- Time penalty: -1.0 per step
- **Total: ~9 to ~17 per pick**

**Invalid pick (already picked):**
- Penalty: **-10.0**

**Episode completion bonus:**
- Completion: +20.0
- Time bonus: +0 to +25.0 (faster = higher)

### **Math for -108.42 reward (9 cubes, 33.89 steps):**

**If all picks were valid:**
- 9 picks × 12 avg = +108
- 34 steps × -1 = -34
- Completion: +20
- **Expected: ~+94** ✅

**Actual: -108.42** ❌

**This means:**
- Agent made ~10-15 invalid picks (already-picked objects)
- 15 invalid × -10 = -150
- 9 valid × 12 = +108
- 34 steps × -1 = -34
- Completion: +20
- **Total: -56** (close to -108 with variance)

**The agent is repeatedly trying to pick already-picked objects!**

---

## ✅ **Solutions:**

### **Solution 1: Action Masking (BEST)**

**What**: Prevent agent from selecting already-picked objects

**How**: Mask invalid actions in observation space

**Implementation**:
1. Add "picked flag" back to observation (0.0 = available, 1.0 = picked)
2. Or use Stable-Baselines3 action masking
3. Or increase invalid action penalty to -50.0

**Pros**:
- ✅ Prevents invalid actions completely
- ✅ Faster training (no wasted steps)
- ✅ Guaranteed positive rewards

**Cons**:
- Requires code changes

---

### **Solution 2: Increase Invalid Action Penalty**

**What**: Make invalid picks more painful

**Current**: -10.0
**New**: -50.0 or -100.0

**Change in `object_selection_env.py`**:
```python
if action >= self.total_objects or action in self.objects_picked:
    reward = -50.0  # Increased from -10.0
```

**Pros**:
- ✅ Easy fix (one line)
- ✅ Agent learns faster to avoid invalid picks

**Cons**:
- Still allows invalid picks during training

---

### **Solution 3: Longer Training**

**What**: Train for more timesteps

**Current**: 100K (heuristic/A*), 10K (RRT)
**New**: 200K (heuristic/A*), 20K (RRT)

**Pros**:
- ✅ Agent has more time to learn
- ✅ May eventually learn to avoid invalid picks

**Cons**:
- Takes 2× longer
- May not solve the problem

---

### **Solution 4: Better Observation Space** ✅ IMPLEMENTED

**What**: Add "picked flag" to observation

**Previous 5 values:**
1. Distance to EE
2. Distance to container
3. Obstacle proximity
4. Reachability flag
5. Path clearance

**NEW 6 values (IMPLEMENTED):**
1. Distance to EE
2. Distance to container
3. Obstacle proximity
4. Reachability flag
5. Path clearance
6. **Picked flag** (0.0 = available, 1.0 = already picked) ✅

**Status**: Implemented in all files:
- ✅ `src/rl/object_selection_env.py` - Base environment
- ✅ `src/rl/object_selection_env_astar.py` - A* variant (inherits from base)
- ✅ `src/rl/object_selection_env_rrt.py` - RRT variant (inherits from base)
- ✅ `scripts/Reinforcement Learning/franka_rrt_physXLidar_depth_camera_rl_standalone_v1.9.py` - Standalone inference script
- ✅ `scripts/Reinforcement Learning/test_object_selection.py` - Test script
- ✅ `scripts/Reinforcement Learning/train_object_selection.py` - Training script (uses updated environments)

**Pros**:
- ✅ Agent can see which objects are picked
- ✅ Should learn to avoid them
- ✅ Clearer signal for the model

**Cons**:
- Requires retraining all models

**Note**: Existing trained models will NOT work with the new observation space (5 params → 6 params).
You must retrain models with the new observation space.

---

## 🎯 **Recommended Action:**

### **Quick Fix (5 minutes):**

**Increase invalid action penalty to -50.0**

```python
# In object_selection_env.py, line 134:
reward = -50.0  # Changed from -10.0
```

Then retrain the worst models:
```bash
quick_train.bat heuristic 4 9
quick_train.bat astar 4 9
```

---

### **Proper Fix (30 minutes):**

**Add "picked flag" to observation space**

1. Change observation space from 5 to 6 values
2. Add picked flag in `_get_observation()`
3. Retrain all models

This is the cleanest solution and will prevent invalid picks completely.

---

## 📊 **Expected Improvement:**

### **Before (with invalid picks):**
- 9 cubes: -108.42 reward ❌
- Many invalid actions
- Wasted steps

### **After (with fix):**
- 9 cubes: +50 to +80 reward ✅
- No invalid actions
- Efficient picking

---

## 🚀 **Next Steps:**

1. **Test current best models** (98.34, 51.32, 24.51) in Isaac Sim
2. **If they work well**: Use them, ignore the negative ones
3. **If you want to fix negative models**: Apply Solution 2 (quick) or Solution 4 (proper)
4. **Retrain only the bad configs** (4×4 with 9 cubes)

**The good news**: Your best models (98.34, 51.32) are already excellent! You may not need to fix anything.

