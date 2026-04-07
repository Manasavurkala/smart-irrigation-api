"""
Smart Irrigation System - Q-Learning RL Model Training
Dataset: irrigation_prediction_1_csv.xlsx (10,000 records)

State Space: 18 states (3 soil moisture x 3 temperature x 2 rain)
Action Space: 3 actions (No Water / 5 Min Water / 10 Min Water)

Dataset mapping:
  Soil Moisture:   <25%  → Low    |  25-50% → Medium  |  >50% → High
  Temperature:     <20°C → Cool   |  20-35°C → Normal  |  >35°C → Hot
  Rainfall_mm:     >500mm → Rain  |  <=500mm → No Rain
  Irrigation_Need: Low → No Water | Medium → 5 Min Water | High → 10 Min Water
"""

import numpy as np
import pandas as pd
import random
import os

# ── Load and preprocess dataset ───────────────────────────────────────────────
df = pd.read_excel('irrigation_prediction 1.csv.xlsx')

print(f"Dataset loaded: {df.shape[0]} records")
print(f"Columns: {df.columns.tolist()}")
print(f"\nIrrigation_Need distribution:\n{df['Irrigation_Need'].value_counts()}")

# ── Discretize continuous values into categories matching project design ──────

def get_moisture_level(moisture):
    """
    Low:    < 25%   → Dry soil, needs water
    Medium: 25-50%  → Optimal range
    High:   > 50%   → Waterlogged, do not water
    """
    if moisture < 25:
        return 0   # Low
    elif moisture <= 50:
        return 1   # Medium
    else:
        return 2   # High

def get_temp_level(temp):
    """
    Cool:   < 20°C  → Low evaporation
    Normal: 20-35°C → Moderate evaporation
    Hot:    > 35°C  → High evaporation, more water needed
    """
    if temp < 20:
        return 0   # Cool
    elif temp <= 35:
        return 1   # Normal
    else:
        return 2   # Hot

def get_rain_status(rainfall_mm):
    """
    Rain:    > 500mm annual → Rain present
    No Rain: <= 500mm       → No rain
    Note: Dataset uses annual rainfall. >500mm means rainy conditions.
    """
    return 1 if rainfall_mm > 500 else 0

def encode_state(moisture_level, temp_level, rain):
    """Encode (3 x 3 x 2) = 18 states as integer 0-17"""
    return moisture_level * 6 + temp_level * 2 + rain

def decode_state(state):
    """Decode state integer back to (moisture_level, temp_level, rain)"""
    moisture_level = state // 6
    remainder = state % 6
    temp_level = remainder // 2
    rain = remainder % 2
    return moisture_level, temp_level, rain

def irrigation_need_to_action(irrigation_need):
    """
    Map dataset labels to RL actions:
    Low    → Action 0: No Water
    Medium → Action 1: 5 Min Water
    High   → Action 2: 10 Min Water
    """
    mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    return mapping.get(irrigation_need, 1)

# ── Process dataset ───────────────────────────────────────────────────────────
df['moisture_level'] = df['Soil_Moisture'].apply(get_moisture_level)
df['temp_level'] = df['Temperature_C'].apply(get_temp_level)
df['rain'] = df['Rainfall_mm'].apply(get_rain_status)
df['state'] = df.apply(lambda r: encode_state(r['moisture_level'], r['temp_level'], r['rain']), axis=1)
df['action'] = df['Irrigation_Need'].apply(irrigation_need_to_action)

print(f"\n--- Discretized Distribution ---")
print(f"Moisture Levels:  Low={sum(df['moisture_level']==0)}, Medium={sum(df['moisture_level']==1)}, High={sum(df['moisture_level']==2)}")
print(f"Temp Levels:      Cool={sum(df['temp_level']==0)}, Normal={sum(df['temp_level']==1)}, Hot={sum(df['temp_level']==2)}")
print(f"Rain Status:      No Rain={sum(df['rain']==0)}, Rain={sum(df['rain']==1)}")
print(f"Actions:          No Water={sum(df['action']==0)}, 5Min={sum(df['action']==1)}, 10Min={sum(df['action']==2)}")
print(f"States covered:   {df['state'].nunique()} / 18")

# ── Reward Function ───────────────────────────────────────────────────────────
def compute_reward(moisture_level, temp_level, rain, action):
    """
    Sustainability-oriented reward function:
    +10  → Optimal decision: medium moisture maintained
    -10  → Dry soil left without water
    -8   → Waterlogged soil given more water
    -5   → Watering when rain is present (wasteful)
    +3   → Correct conservation (no water when not needed)
    -3   → Under-watering when soil is dry
    """
    reward = 0

    # Penalize watering when rain is present (wasteful)
    if rain == 1 and action > 0:
        reward -= 5

    # Dry soil logic
    if moisture_level == 0:  # Low moisture - dry soil
        if action == 2:      # 10 min water - correct for very dry
            reward += 10
        elif action == 1:    # 5 min water - acceptable
            reward += 5
        else:                # No water - bad, soil stays dry
            reward -= 10

    # Optimal moisture logic
    elif moisture_level == 1:  # Medium moisture - optimal range
        if temp_level == 2:    # Hot temperature - needs some water to compensate evaporation
            if action == 1:
                reward += 10
            elif action == 0:
                reward += 3
            else:
                reward -= 3
        else:                  # Cool/Normal temp - soil is fine
            if action == 0:
                reward += 10   # Best: conserve water
            elif action == 1:
                reward += 2
            else:
                reward -= 5    # Over-watering

    # Waterlogged soil logic
    elif moisture_level == 2:  # High moisture - soil is wet
        if action == 0:
            reward += 10       # Correct: don't add more water
        elif action == 1:
            reward -= 5        # Mild over-watering
        else:
            reward -= 8        # Severe over-watering

    return reward

# ── Q-Learning Training ───────────────────────────────────────────────────────
N_STATES  = 18
N_ACTIONS = 3
ALPHA     = 0.1    # Learning rate
GAMMA     = 0.95   # Discount factor
EPSILON   = 1.0    # Initial exploration rate
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01
EPISODES  = 15000

q_table = np.zeros((N_STATES, N_ACTIONS))
episode_rewards = []

# Convert dataset to list of (state, action, moisture_level, temp_level, rain) tuples
dataset_samples = list(zip(
    df['state'], df['action'], df['moisture_level'], df['temp_level'], df['rain']
))

print(f"\n--- Starting Q-Learning Training ---")
print(f"Episodes: {EPISODES}, Alpha: {ALPHA}, Gamma: {GAMMA}")
print(f"State Space: {N_STATES}, Action Space: {N_ACTIONS}")

for episode in range(EPISODES):
    # Sample a random row from the dataset as starting state
    sample = random.choice(dataset_samples)
    state, dataset_action, moisture_level, temp_level, rain = sample

    total_reward = 0

    # Epsilon-greedy action selection
    if random.random() < EPSILON:
        # Explore: sometimes use dataset action, sometimes random
        if random.random() < 0.6:
            action = dataset_action   # Learn from dataset
        else:
            action = random.randint(0, N_ACTIONS - 1)
    else:
        action = np.argmax(q_table[state])  # Exploit

    # Compute reward
    reward = compute_reward(moisture_level, temp_level, rain, action)
    total_reward += reward

    # Simulate next state (soil moisture changes after irrigation)
    # After watering, moisture level may increase
    next_moisture = moisture_level
    if action == 2 and moisture_level < 2:
        next_moisture = min(2, moisture_level + 1)
    elif action == 0 and moisture_level > 0 and temp_level == 2:
        next_moisture = max(0, moisture_level - 1)   # Hot temp dries soil
    next_rain = random.choice([0, 1]) if rain == 0 else random.choice([0, 0, 1])
    next_state = encode_state(next_moisture, temp_level, next_rain)

    # Bellman equation update
    best_next = np.max(q_table[next_state])
    q_table[state, action] += ALPHA * (reward + GAMMA * best_next - q_table[state, action])

    # Decay epsilon
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    if (episode + 1) % 3000 == 0:
        avg_reward = np.mean(episode_rewards[-3000:])
        print(f"Episode {episode+1}/{EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {EPSILON:.4f}")

# ── Save Q-Table ──────────────────────────────────────────────────────────────
os.makedirs('/home/claude/smart_irrigation/models', exist_ok=True)
np.save('/home/claude/smart_irrigation/models/q_table.npy', q_table)
print(f"\n--- Q-Table saved to models/q_table.npy ---")

# ── Print Final Q-Table ───────────────────────────────────────────────────────
action_names = ['No Water', '5 Min Water', '10 Min Water']
moisture_names = ['Low (<25%)', 'Medium (25-50%)', 'High (>50%)']
temp_names = ['Cool (<20°C)', 'Normal (20-35°C)', 'Hot (>35°C)']
rain_names = ['No Rain', 'Rain']

print("\n--- Final Q-Table (Optimal Actions for All 18 States) ---")
print(f"{'State':>5} | {'Soil Moisture':>15} | {'Temperature':>15} | {'Rain':>8} | {'Best Action':>12} | Q-Values")
print("-" * 90)
for state in range(N_STATES):
    m, t, r = decode_state(state)
    best_action = np.argmax(q_table[state])
    q_vals = [f"{v:.2f}" for v in q_table[state]]
    print(f"  {state:>3} | {moisture_names[m]:>15} | {temp_names[t]:>15} | {rain_names[r]:>8} | {action_names[best_action]:>12} | {q_vals}")

print("\n--- Training Complete ---")
print(f"Final Epsilon: {EPSILON:.4f}")
print(f"Average reward (last 1000 episodes): {np.mean(episode_rewards[-1000:]):.2f}")
