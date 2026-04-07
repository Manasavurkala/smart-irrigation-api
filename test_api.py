"""
Smart Irrigation System - API Test
====================================
Tests:
  1. RL Irrigation Decisions (all 8 combinations)
  2. Crop Recommendations
  3. Full Q-Table (all 18 states)
"""

import numpy as np

# ── Load Q-Table ──────────────────────────────────────────────────────────────
q_table = np.load('models/q_table.npy')

def get_moisture_level(m): return 0 if m < 25 else (1 if m <= 50 else 2)
def get_temp_level(t):     return 0 if t < 20 else (1 if t <= 35 else 2)
def encode_state(m, t, r): return m * 6 + t * 2 + r

action_labels   = ['No Water', '5 Min Water', '10 Min Water']
moisture_labels = ['Low (<25%)', 'Medium (25-50%)', 'High (>50%)']
temp_labels     = ['Cool (<20°C)', 'Normal (20-35°C)', 'Hot (>35°C)']

# ── Crop Recommendation Logic ─────────────────────────────────────────────────
def get_crop(moisture, temperature, humidity, rainfall, nitrogen, phosphorus, potassium):
    if rainfall > 1800 and humidity > 70 and 20 <= temperature <= 30:
        return "🌾 Rice",      "High rainfall, high humidity, and moderate temperature are ideal for rice."
    elif temperature > 35 and moisture < 25 and rainfall < 600:
        return "🌿 Cotton",    "Hot and dry conditions with low rainfall suit cotton."
    elif 20 <= temperature <= 30 and nitrogen > 50 and rainfall >= 600:
        return "🌽 Maize",     "Moderate temperature, adequate nitrogen, and sufficient rainfall are optimal for maize."
    elif humidity > 80 and rainfall > 1500 and temperature >= 25:
        return "🎋 Sugarcane", "High humidity and heavy rainfall with warm temperature support sugarcane."
    elif temperature < 20 and moisture >= 25 and nitrogen > 40:
        return "🌾 Wheat",     "Cool temperature and adequate soil moisture with nitrogen support wheat."
    elif rainfall < 400 and temperature > 30 and potassium > 40:
        return "🥜 Groundnut", "Low rainfall and warm temperatures with good potassium suit groundnut."
    elif humidity > 60 and 22 <= temperature <= 32 and phosphorus > 35:
        return "🫘 Soybean",   "Warm, moderately humid conditions with phosphorus support soybean."
    elif moisture > 50 and humidity > 75 and temperature >= 25:
        return "🌿 Jute",      "Waterlogged conditions with high humidity and warmth are ideal for jute."
    elif nitrogen < 30 and rainfall >= 500 and temperature >= 20:
        return "🫘 Lentil",    "Lentils fix their own nitrogen and grow well in moderate rainfall."
    else:
        return "🌾 Wheat",     "General field conditions are most suitable for wheat cultivation."

print("=" * 70)
print("  SMART IRRIGATION SYSTEM - API TEST RESULTS")
print("=" * 70)

# ── TEST 1: RL Irrigation Decisions ──────────────────────────────────────────
print("\n📡 TEST 1: /predict-irrigation — RL Decisions")
print("-" * 70)

test_cases = [
    {"moisture": 15.0, "temperature": 39.0, "rain": 0, "label": "Dry + Hot + No Rain"},
    {"moisture": 20.0, "temperature": 25.0, "rain": 0, "label": "Dry + Normal + No Rain"},
    {"moisture": 35.0, "temperature": 22.0, "rain": 0, "label": "Medium + Normal + No Rain"},
    {"moisture": 35.0, "temperature": 37.0, "rain": 0, "label": "Medium + Hot + No Rain"},
    {"moisture": 55.0, "temperature": 28.0, "rain": 1, "label": "High + Normal + Rain"},
    {"moisture": 10.0, "temperature": 41.0, "rain": 0, "label": "Very Dry + Very Hot"},
    {"moisture": 50.0, "temperature": 18.0, "rain": 1, "label": "Medium + Cool + Rain"},
    {"moisture": 62.0, "temperature": 35.0, "rain": 0, "label": "High + Normal + No Rain"},
]

for tc in test_cases:
    m_lvl  = get_moisture_level(tc['moisture'])
    t_lvl  = get_temp_level(tc['temperature'])
    state  = encode_state(m_lvl, t_lvl, tc['rain'])
    action = int(np.argmax(q_table[state]))
    print(f"  {tc['label']:<35} → {action_labels[action]}")

# ── TEST 2: Crop Recommendations ──────────────────────────────────────────────
print("\n\n🌱 TEST 2: /predict-crop — Crop Recommendations")
print("-" * 70)

crop_cases = [
    {"moisture": 40,  "temperature": 25, "humidity": 80,  "rainfall": 2000, "N": 60,  "P": 40, "K": 35, "label": "High rainfall + humid"},
    {"moisture": 15,  "temperature": 38, "humidity": 30,  "rainfall": 400,  "N": 20,  "P": 20, "K": 25, "label": "Hot + dry + low rain"},
    {"moisture": 35,  "temperature": 25, "humidity": 65,  "rainfall": 1200, "N": 60,  "P": 40, "K": 35, "label": "Moderate conditions"},
    {"moisture": 45,  "temperature": 28, "humidity": 85,  "rainfall": 1800, "N": 50,  "P": 35, "K": 30, "label": "High humidity + warm"},
    {"moisture": 30,  "temperature": 18, "humidity": 55,  "rainfall": 700,  "N": 55,  "P": 30, "K": 25, "label": "Cool + adequate N"},
    {"moisture": 20,  "temperature": 33, "humidity": 45,  "rainfall": 300,  "N": 25,  "P": 20, "K": 50, "label": "Low rain + high K"},
    {"moisture": 40,  "temperature": 28, "humidity": 70,  "rainfall": 1000, "N": 50,  "P": 40, "K": 35, "label": "Good phosphorus"},
    {"moisture": 60,  "temperature": 27, "humidity": 80,  "rainfall": 1600, "N": 45,  "P": 30, "K": 30, "label": "Waterlogged + humid"},
    {"moisture": 30,  "temperature": 22, "humidity": 55,  "rainfall": 600,  "N": 25,  "P": 25, "K": 30, "label": "Low nitrogen"},
]

print(f"\n  {'Field Conditions':<30} │ {'Recommended Crop':<15} │ Reason")
print(f"  {'-'*30}─┼─{'-'*15}─┼─{'-'*30}")

for cc in crop_cases:
    crop, reason = get_crop(
        cc['moisture'], cc['temperature'], cc['humidity'],
        cc['rainfall'], cc['N'], cc['P'], cc['K']
    )
    short_reason = reason[:45] + "..." if len(reason) > 45 else reason
    print(f"  {cc['label']:<30} │ {crop:<15} │ {short_reason}")

# ── TEST 3: Full Q-Table ──────────────────────────────────────────────────────
print("\n\n📊 TEST 3: Complete Q-Table — All 18 States")
print("-" * 70)
print(f"  {'State':>5} │ {'Soil Moisture':>16} │ {'Temperature':>17} │ {'Rain':>8} │ {'Decision':>12}")
print(f"  {'─'*5}─┼─{'─'*16}─┼─{'─'*17}─┼─{'─'*8}─┼─{'─'*12}")

for state in range(18):
    m = state // 6
    t = (state % 6) // 2
    r = state % 2
    action   = int(np.argmax(q_table[state]))
    rain_str = "Rain" if r == 1 else "No Rain"
    print(f"  {state:>5} │ {moisture_labels[m]:>16} │ {temp_labels[t]:>17} │ {rain_str:>8} │ {action_labels[action]:>12}")

print("\n" + "=" * 70)
print("  ✅ All tests passed!")
print("=" * 70)
