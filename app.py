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
        print("⚠  MONGO_URI not set in .env — MongoDB disabled.")
        return
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command("ping")
        db = mongo_client[DB_NAME]
        mongo_ok = True
        print(f"✅ MongoDB Atlas connected → database: '{DB_NAME}'")
        _create_indexes()
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("   App will still run — data will NOT be saved to Atlas.")

def _create_indexes():
    db.sensor_readings.create_index([("timestamp", DESCENDING)])
    db.rl_decisions.create_index([("timestamp", DESCENDING)])
    db.pump_logs.create_index([("timestamp", DESCENDING)])
    print("✅ MongoDB indexes created.")

connect_mongodb()

# ── Load Q-Table ─────────────────────────────────────
try:
    q_table = np.load("models/q_table.npy")
    print("✅ Q-Table loaded:", q_table.shape)
except FileNotFoundError:
    print("⚠  Q-Table not found. Run train_rl.py first.")
    q_table = np.zeros((18, 3))

# ── In-memory state ───────────────────────────────────
sensor_lock = threading.Lock()
latest_sensor = {
    "moisture": 36.97, "temperature": 26.99,
    "humidity": 60.08,  "rainfall": 1252.5,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "source": "default",
}
pump_state = {"on": False, "action_label": "No Water", "duration_minutes": 0}

# ── MongoDB helpers ──────────────────────────────────
def save_sensor_reading(doc: dict):
    if not mongo_ok:
        return
    try:
        db.sensor_readings.insert_one(doc)
    except Exception as e:
        print(f"⚠  MongoDB write error (sensor_readings): {e}")

def save_rl_decision(doc: dict):
    if not mongo_ok:
        return
    try:
        db.rl_decisions.insert_one(doc)
    except Exception as e:
        print(f"⚠  MongoDB write error (rl_decisions): {e}")

def save_pump_log(doc: dict):
    if not mongo_ok:
        return
    try:
        db.pump_logs.insert_one(doc)
    except Exception as e:
        print(f"⚠  MongoDB write error (pump_logs): {e}")

def strip_id(doc):
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

# ── RL helpers ───────────────────────────────────────
def get_moisture_level(m): return 0 if m < 25 else (1 if m <= 50 else 2)
def get_temp_level(t): return 0 if t < 20 else (1 if t <= 35 else 2)
def encode_state(ml, tl, rain): return ml * 6 + tl * 2 + rain
def rl_explanation(ml, tl, rain, action):
    ml_map  = {0: "low (dry soil)", 1: "optimal", 2: "high (wet soil)"}
    tl_map  = {0: "cool", 1: "normal", 2: "hot"}
    rain_str = "rain detected" if rain else "no rain"
    act_map  = {0: "No irrigation needed", 1: "5-minute irrigation", 2: "10-minute irrigation"}
    return f"{act_map[action]} — soil moisture is {ml_map[ml]}, temperature is {tl_map[tl]}, {rain_str}."

# ── Crop recommendation ──────────────────────────────
def recommend_crop(moisture, temperature, humidity, rainfall):
    if rainfall > 1500 and humidity > 70 and 20 <= temperature <= 35:
        return "Rice 🌾", "High rainfall + high humidity + moderate temp"
    if temperature > 32 and moisture < 25 and rainfall < 600:
        return "Cotton 🌿", "Hot temp + dry soil + low rainfall"
    if 22 <= temperature <= 32 and 25 <= moisture <= 50 and 600 <= rainfall <= 1500:
        return "Maize 🌽", "Moderate temp + optimal moisture + medium rainfall"
    if humidity > 75 and rainfall > 1400 and temperature >= 26:
        return "Sugarcane 🎋", "High humidity + abundant rainfall + warm temp"
    if temperature < 22 and moisture >= 25 and 300 <= rainfall <= 1200:
        return "Wheat 🌾", "Cool temp + adequate moisture + moderate rainfall"
    if temperature > 28 and humidity < 50 and rainfall < 700:
        return "Groundnut 🥜", "Warm temp + low humidity + limited rainfall"
    if 50 <= humidity <= 75 and 22 <= temperature <= 32 and moisture >= 25:
        return "Soybean 🫘", "Moderate humidity + warm temp + good soil moisture"
    if moisture > 50 and humidity > 70 and temperature >= 25 and rainfall > 1200:
        return "Jute 🌿", "Waterlogged soil + very high humidity + warm temp"
    if temperature < 22 and humidity > 55 and rainfall >= 300:
        return "Mustard 🌻", "Cool temp + moderate humidity + rainfall"
    if moisture < 25 and temperature > 28 and 300 <= rainfall <= 1000:
        return "Millet (Bajra) 🌾", "Low moisture + high temp + limited rainfall"
    return "Wheat 🌾", "Conditions broadly suitable for wheat"

# ── ROUTES ──────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Smart Irrigation API is running! Use /predict, /latest-reading, etc.",
        "endpoints": [
            "/predict",
            "/sensor-data",
            "/latest-reading",
            "/pump-state",
            "/history/sensors",
            "/history/decisions",
            "/history/crops",
            "/stats",
            "/health"
        ]
    })

# Keep all your other routes (predict, sensor-data, latest-reading, etc.) as-is…

# ── RUN APP ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)