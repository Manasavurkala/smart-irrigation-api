"""
Smart Irrigation System — Flask Backend + MongoDB Atlas
========================================================
All 4 sensor readings, RL decisions, and crop recommendations
are stored in MongoDB Atlas automatically.
"""

import os
import threading
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# ── Load .env ─────────────────────────────────────────
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME   = os.getenv("DB_NAME", "smart_irrigation")

# ── Flask ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── MongoDB Atlas connection ──────────────────────────
mongo_client = None
db           = None
mongo_ok     = False

def connect_mongodb():
    global mongo_client, db, mongo_ok
    if not MONGO_URI:
        print("⚠ MONGO_URI not set — MongoDB disabled.")
        return
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command("ping")
        db = mongo_client[DB_NAME]
        mongo_ok = True
        print(f"✅ MongoDB connected → '{DB_NAME}'")
        _create_indexes()
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"❌ MongoDB connection failed: {e}")

def _create_indexes():
    db.sensor_readings.create_index([("timestamp", DESCENDING)])
    db.rl_decisions.create_index([("timestamp", DESCENDING)])
    db.pump_logs.create_index([("timestamp", DESCENDING)])

connect_mongodb()

# ── Load Q-Table ─────────────────────────────────────
try:
    q_table = np.load("models/q_table.npy")
except:
    q_table = np.zeros((18, 3))

# ── In-memory state ───────────────────────────────────
sensor_lock = threading.Lock()
latest_sensor = {}
pump_state = {"on": False, "action_label": "No Water", "duration_minutes": 0}

# ── RL helpers ───────────────────────────────────────
def get_moisture_level(m): return 0 if m < 25 else (1 if m <= 50 else 2)
def get_temp_level(t): return 0 if t < 20 else (1 if t <= 35 else 2)
def encode_state(ml, tl, rain): return ml * 6 + tl * 2 + rain

# ── ROUTES ──────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Smart Irrigation API is running!",
        "endpoints": [
            "/predict",
            "/sensor-data",
            "/latest-reading",
            "/pump-state",
            "/history/sensors",
            "/history/decisions",
            "/stats",
            "/health"
        ]
    })

# ✅ HEALTH CHECK
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "mongodb_connected": mongo_ok
    })

# ✅ RECEIVE SENSOR DATA FROM ESP32
@app.route("/sensor-data", methods=["POST"])
def receive_sensor_data():
    global latest_sensor
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    required_fields = ["moisture", "temperature", "humidity", "rainfall"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    with sensor_lock:
        latest_sensor = {
            "moisture": float(data["moisture"]),
            "temperature": float(data["temperature"]),
            "humidity": float(data["humidity"]),
            "rainfall": float(data["rainfall"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "esp32"
        }

    if mongo_ok:
        db.sensor_readings.insert_one(latest_sensor)

    return jsonify({"message": "Sensor data received successfully"})

# ✅ GET LATEST SENSOR READING
@app.route("/latest-reading")
def latest_reading():
    if not mongo_ok:
        return jsonify({"error": "MongoDB not connected"}), 500

    latest = db.sensor_readings.find_one(sort=[("timestamp", -1)])

    if not latest:
        return jsonify({})

    latest["_id"] = str(latest["_id"])
    return jsonify(latest)
# ✅ RL PREDICTION
@app.route("/predict", methods=["POST"])
def predict():
    global pump_state

    if not latest_sensor:
        return jsonify({"error": "No sensor data available"}), 400

    moisture = latest_sensor["moisture"]
    temperature = latest_sensor["temperature"]
    humidity = latest_sensor["humidity"]
    rainfall_raw = latest_sensor["rainfall"]
    rainfall = 1 if rainfall_raw > 500 else 0

    ml = get_moisture_level(moisture)
    tl = get_temp_level(temperature)
    state = encode_state(ml, tl, rainfall)

    q_values = q_table[state]
    action = int(np.argmax(q_values))

    action_map = {
        0: "No Water",
        1: "5 Min Water",
        2: "10 Min Water"
    }

    decision_label = action_map[action]

    irrigation_on = action != 0

    explanation = f"State {state} → Moisture level {ml}, Temp level {tl}, Rain flag {rainfall}. Best action selected from trained Q-table."

    # Simple crop recommendation logic
    if moisture > 40 and rainfall_raw > 1000:
        crop = "Rice 🌾"
        crop_reason = "High moisture and high rainfall are ideal for rice."
    elif temperature > 30 and humidity < 50:
        crop = "Cotton 🌿"
        crop_reason = "Hot and moderate humidity conditions suit cotton."
    else:
        crop = "Maize 🌽"
        crop_reason = "Balanced conditions suitable for maize."

    pump_state = {
        "on": irrigation_on,
        "action_label": decision_label,
        "duration_minutes": 0 if action == 0 else (5 if action == 1 else 10),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if mongo_ok:
        db.rl_decisions.insert_one({
            "state": state,
            "action": action,
            "decision": decision_label,
            "timestamp": datetime.now()
        })

    return jsonify({
        "action": action,
        "decision": decision_label,
        "explanation": explanation,
        "state": state,
        "irrigation_on": irrigation_on,
        "q_values": {
            "No Water": float(q_values[0]),
            "5 Min Water": float(q_values[1]),
            "10 Min Water": float(q_values[2])
        },
        "crop": crop,
        "crop_reason": crop_reason
    })
# ✅ PUMP STATE
@app.route("/pump-state", methods=["GET"])
def get_pump_state():
    return jsonify(pump_state)

# ✅ SENSOR HISTORY
@app.route("/history/sensors", methods=["GET"])
def sensor_history():
    if not mongo_ok:
        return jsonify({"error": "MongoDB not connected"}), 500
    data = list(db.sensor_readings.find().sort("timestamp", -1).limit(50))
    for d in data:
        d["_id"] = str(d["_id"])
    return jsonify(data)

# ✅ RL DECISION HISTORY
@app.route("/history/decisions", methods=["GET"])
def decision_history():
    if not mongo_ok:
        return jsonify({"error": "MongoDB not connected"}), 500
    data = list(db.rl_decisions.find().sort("timestamp", -1).limit(50))
    for d in data:
        d["_id"] = str(d["_id"])
    return jsonify(data)

# ✅ BASIC STATS
@app.route("/stats", methods=["GET"])
def stats():
    if not mongo_ok:
        return jsonify({"error": "MongoDB not connected"}), 500

    total_sensors = db.sensor_readings.count_documents({})
    total_decisions = db.rl_decisions.count_documents({})

    return jsonify({
        "total_sensor_readings": total_sensors,
        "total_rl_decisions": total_decisions
    })

# ── RUN APP ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)