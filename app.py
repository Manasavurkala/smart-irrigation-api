"""
Smart Irrigation System — Flask Backend + MongoDB Atlas
========================================================
All 4 sensor readings, RL decisions, and crop recommendations
are stored in MongoDB Atlas automatically.

Collections in MongoDB:
  sensor_readings   → every reading from ESP32 / Node-RED
  rl_decisions      → every RL prediction + crop recommendation
  pump_logs         → every time pump turns ON or OFF

Setup:
  1. pip install flask flask-cors numpy pymongo python-dotenv
  2. Create a .env file with:
       MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
       DB_NAME=smart_irrigation
  3. python app.py
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

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME   = os.getenv("DB_NAME",   "smart_irrigation")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── MongoDB Atlas connection ──────────────────────────────────────────────────
mongo_client = None
db           = None
mongo_ok     = False

def connect_mongodb():
    """Try to connect to MongoDB Atlas. Sets mongo_ok = True on success."""
    global mongo_client, db, mongo_ok
    if not MONGO_URI:
        print("⚠  MONGO_URI not set in .env — MongoDB disabled.")
        return
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command("ping")   # test connection
        db       = mongo_client[DB_NAME]
        mongo_ok = True
        print(f"✅ MongoDB Atlas connected → database: '{DB_NAME}'")
        _create_indexes()
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("   App will still run — data will NOT be saved to Atlas.")

def _create_indexes():
    """Create indexes for faster queries."""
    db.sensor_readings.create_index([("timestamp", DESCENDING)])
    db.rl_decisions.create_index([("timestamp",    DESCENDING)])
    db.pump_logs.create_index([("timestamp",        DESCENDING)])
    print("✅ MongoDB indexes created.")

connect_mongodb()

# ── Load Q-Table ──────────────────────────────────────────────────────────────
try:
    q_table = np.load("models/q_table.npy")
    print("✅ Q-Table loaded:", q_table.shape)
except FileNotFoundError:
    print("⚠  Q-Table not found. Run train_rl.py first.")
    q_table = np.zeros((18, 3))

# ── In-memory state (fallback when MongoDB is down) ──────────────────────────
sensor_lock   = threading.Lock()
latest_sensor = {
    "moisture": 36.97, "temperature": 26.99,
    "humidity": 60.08,  "rainfall": 1252.5,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "source": "default",
}
pump_state = {"on": False, "action_label": "No Water", "duration_minutes": 0}


# ── MongoDB helpers ───────────────────────────────────────────────────────────
def save_sensor_reading(doc: dict):
    """Save one sensor reading to MongoDB. Silently skips if Atlas is down."""
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
    """Remove MongoDB _id field so jsonify doesn't choke on ObjectId."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


# ── RL helpers ────────────────────────────────────────────────────────────────
def get_moisture_level(m):
    return 0 if m < 25 else (1 if m <= 50 else 2)

def get_temp_level(t):
    return 0 if t < 20 else (1 if t <= 35 else 2)

def encode_state(ml, tl, rain):
    return ml * 6 + tl * 2 + rain

def rl_explanation(ml, tl, rain, action):
    ml_map  = {0: "low (dry soil)", 1: "optimal", 2: "high (wet soil)"}
    tl_map  = {0: "cool", 1: "normal", 2: "hot"}
    rain_str = "rain detected" if rain else "no rain"
    act_map  = {
        0: "No irrigation needed",
        1: "5-minute irrigation",
        2: "10-minute irrigation",
    }
    return (
        f"{act_map[action]} — soil moisture is {ml_map[ml]}, "
        f"temperature is {tl_map[tl]}, {rain_str}."
    )


# ── Crop recommendation ───────────────────────────────────────────────────────
def recommend_crop(moisture, temperature, humidity, rainfall):
    """
    Rules derived from your 4 dataset columns only (no N/P/K).
    Mirrors the client-side logic in smart_irrigation_app.jsx exactly.
    """
    if rainfall > 1500 and humidity > 70 and 20 <= temperature <= 35:
        return ("Rice 🌾",
                "High annual rainfall (>1500mm), high humidity, and moderate temperature "
                "perfectly match rice cultivation. Rice is the top Kharif crop in India.")

    if temperature > 32 and moisture < 25 and rainfall < 600:
        return ("Cotton 🌿",
                "Hot temperature, dry soil, and low rainfall are ideal for cotton — "
                "a drought-tolerant cash crop that thrives in Telangana and Gujarat.")

    if 22 <= temperature <= 32 and 25 <= moisture <= 50 and 600 <= rainfall <= 1500:
        return ("Maize 🌽",
                "Moderate temperature, optimal soil moisture, and medium rainfall create "
                "excellent conditions for maize — a versatile Kharif and Rabi crop.")

    if humidity > 75 and rainfall > 1400 and temperature >= 26:
        return ("Sugarcane 🎋",
                "Very high humidity, abundant rainfall, and warm temperature support "
                "high-yield sugarcane — a major cash crop in Andhra Pradesh and UP.")

    if temperature < 22 and moisture >= 25 and 300 <= rainfall <= 1200:
        return ("Wheat 🌾",
                "Cool temperature with adequate soil moisture and moderate rainfall are "
                "the classic Rabi conditions for high-yield wheat cultivation.")

    if temperature > 28 and humidity < 50 and rainfall < 700:
        return ("Groundnut 🥜",
                "Warm temperatures, low humidity, and limited rainfall suit groundnut — "
                "drought-resilient and widely grown in Telangana and Andhra Pradesh.")

    if 50 <= humidity <= 75 and 22 <= temperature <= 32 and moisture >= 25:
        return ("Soybean 🫘",
                "Moderately humid, warm conditions with good soil moisture are optimal "
                "for soybean — a high-protein Kharif crop grown across central India.")

    if moisture > 50 and humidity > 70 and temperature >= 25 and rainfall > 1200:
        return ("Jute 🌿",
                "Waterlogged soil, very high humidity, and warm temperature are ideal "
                "for jute fiber cultivation — mainly grown in West Bengal and Assam.")

    if temperature < 22 and humidity > 55 and rainfall >= 300:
        return ("Mustard 🌻",
                "Cool temperatures with moderate humidity and rainfall suit mustard — "
                "a key Rabi oilseed crop grown across Rajasthan, MP, and UP.")

    if moisture < 25 and temperature > 28 and 300 <= rainfall <= 1000:
        return ("Millet (Bajra) 🌾",
                "Low soil moisture, high temperature, and limited rainfall match millet's "
                "drought-tolerant nature — ideal for semi-arid regions of India.")

    return ("Wheat 🌾",
            "Current conditions are broadly suitable for wheat — India's most widely "
            "cultivated Rabi crop, adaptable to a range of soil and weather conditions.")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 1  →  RL decision + crop recommendation  (saved to MongoDB)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST body:
    { "moisture": 35.5, "temperature": 28.0, "humidity": 62.0, "rainfall": 1200.0 }

    Saves result to MongoDB collection: rl_decisions
    Also updates pump_state (ESP32 polls /pump-state to act on it).
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        moisture    = float(data.get("moisture",    36.97))
        temperature = float(data.get("temperature", 26.99))
        humidity    = float(data.get("humidity",    60.08))
        rainfall    = float(data.get("rainfall",  1252.5))
        ts          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── RL decision ───────────────────────────────────────────────────────
        rain  = 1 if rainfall > 500 else 0
        ml    = get_moisture_level(moisture)
        tl    = get_temp_level(temperature)
        state = encode_state(ml, tl, rain)
        q_row = q_table[state]
        action = int(np.argmax(q_row))

        action_labels = ["No Water", "5 Min Water", "10 Min Water"]
        ml_labels     = ["Low (<25%)", "Medium (25-50%)", "High (>50%)"]
        tl_labels     = ["Cool (<20°C)", "Normal (20-35°C)", "Hot (>35°C)"]
        irrigation_on = action > 0

        # Update pump state
        with sensor_lock:
            pump_state.update({
                "on":               irrigation_on,
                "action_label":     action_labels[action],
                "duration_minutes": [0, 5, 10][action],
            })

        # ── Crop recommendation ───────────────────────────────────────────────
        crop, crop_reason = recommend_crop(moisture, temperature, humidity, rainfall)

        # ── Build response payload ────────────────────────────────────────────
        response_data = {
            "action":          action,
            "decision":        action_labels[action],
            "irrigation_on":   irrigation_on,
            "explanation":     rl_explanation(ml, tl, rain, action),
            "state":           state,
            "moisture_level":  ml_labels[ml],
            "temp_level":      tl_labels[tl],
            "rain_detected":   bool(rain),
            "rainfall_input":  rainfall,
            "q_values": {
                "No Water":     round(float(q_row[0]), 2),
                "5 Min Water":  round(float(q_row[1]), 2),
                "10 Min Water": round(float(q_row[2]), 2),
            },
            "pump_state":  pump_state,
            "crop":        crop,
            "crop_reason": crop_reason,
            "timestamp":   ts,
        }

        # ── Save to MongoDB ───────────────────────────────────────────────────
        mongo_doc = {
            "sensor_input": {
                "moisture":    moisture,
                "temperature": temperature,
                "humidity":    humidity,
                "rainfall":    rainfall,
            },
            "rl": {
                "action":        action,
                "decision":      action_labels[action],
                "irrigation_on": irrigation_on,
                "state":         state,
                "explanation":   response_data["explanation"],
                "q_values":      response_data["q_values"],
                "moisture_level": ml_labels[ml],
                "temp_level":     tl_labels[tl],
                "rain_detected":  bool(rain),
            },
            "crop": {
                "name":   crop,
                "reason": crop_reason,
            },
            "pump_state":     dict(pump_state),
            "timestamp":      ts,
            "timestamp_iso":  datetime.now().isoformat(),
        }
        save_rl_decision(mongo_doc)
        print(f"🤖 RL → {action_labels[action]} | Crop: {crop} | Saved to MongoDB: {mongo_ok}")

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 2  →  ESP32 / Node-RED posts sensor data  (saved to MongoDB)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/sensor-data", methods=["POST"])
def receive_sensor_data():
    """
    Called by ESP32 every few seconds.
    Saves reading to MongoDB collection: sensor_readings
    Also updates in-memory latest_sensor (polled by /latest-reading).
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        reading = {
            "moisture":    round(float(data.get("moisture",    36.97)), 2),
            "temperature": round(float(data.get("temperature", 26.99)), 2),
            "humidity":    round(float(data.get("humidity",    60.08)), 2),
            "rainfall":    round(float(data.get("rainfall",  1252.5)), 2),
            "timestamp":   ts,
            "source":      "esp32",
        }

        # Update in-memory state
        with sensor_lock:
            latest_sensor.update(reading)

        # Save to MongoDB
        mongo_doc = {**reading, "timestamp_iso": datetime.now().isoformat()}
        save_sensor_reading(mongo_doc)
        print(f"📡 ESP32 reading saved → MongoDB: {mongo_ok} | {reading}")

        return jsonify({"status": "ok", "received": reading})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 3  →  Frontend polls latest sensor values
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/latest-reading", methods=["GET"])
def latest_reading():
    """Returns the most recent sensor reading (from memory or MongoDB)."""
    # If MongoDB is up, try to get the freshest reading from Atlas
    if mongo_ok:
        try:
            doc = db.sensor_readings.find_one(
                sort=[("timestamp_iso", DESCENDING)]
            )
            if doc:
                return jsonify(strip_id(doc))
        except Exception:
            pass
    # Fallback to in-memory
    with sensor_lock:
        return jsonify(dict(latest_sensor))


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 4  →  ESP32 polls this to control relay
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/pump-state", methods=["GET"])
def get_pump_state():
    """
    ESP32 polls every 3s.
    pump_state.on == true  → open relay → motor runs
    pump_state.on == false → close relay → motor stops
    """
    return jsonify(pump_state)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 5  →  Get history from MongoDB (last N records)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/history/sensors", methods=["GET"])
def history_sensors():
    """Returns last 100 sensor readings from MongoDB."""
    if not mongo_ok:
        return jsonify({"error": "MongoDB not connected", "data": []}), 200
    try:
        limit = int(request.args.get("limit", 100))
        docs  = list(
            db.sensor_readings.find(
                {}, {"_id": 0}
            ).sort("timestamp_iso", DESCENDING).limit(limit)
        )
        return jsonify({"count": len(docs), "data": docs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/decisions", methods=["GET"])
def history_decisions():
    """Returns last 100 RL decisions from MongoDB."""
    if not mongo_ok:
        return jsonify({"error": "MongoDB not connected", "data": []}), 200
    try:
        limit = int(request.args.get("limit", 100))
        docs  = list(
            db.rl_decisions.find(
                {}, {"_id": 0}
            ).sort("timestamp_iso", DESCENDING).limit(limit)
        )
        return jsonify({"count": len(docs), "data": docs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/crops", methods=["GET"])
def history_crops():
    """Returns crop recommendations grouped by crop name."""
    if not mongo_ok:
        return jsonify({"error": "MongoDB not connected", "data": []}), 200
    try:
        pipeline = [
            {"$group": {"_id": "$crop.name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        docs = list(db.rl_decisions.aggregate(pipeline))
        for d in docs:
            d["crop"] = d.pop("_id")
        return jsonify({"data": docs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 6  →  Stats summary from MongoDB
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/stats", methods=["GET"])
def stats():
    """Returns counts and averages from MongoDB."""
    if not mongo_ok:
        return jsonify({"error": "MongoDB not connected"}), 200
    try:
        total_readings  = db.sensor_readings.count_documents({})
        total_decisions = db.rl_decisions.count_documents({})

        # Count by action
        pipeline = [
            {"$group": {"_id": "$rl.decision", "count": {"$sum": 1}}}
        ]
        decision_counts = {d["_id"]: d["count"] for d in db.rl_decisions.aggregate(pipeline)}

        # Average sensor values from all readings
        avg_pipeline = [
            {"$group": {
                "_id": None,
                "avg_moisture":    {"$avg": "$moisture"},
                "avg_temperature": {"$avg": "$temperature"},
                "avg_humidity":    {"$avg": "$humidity"},
                "avg_rainfall":    {"$avg": "$rainfall"},
            }}
        ]
        avg_res  = list(db.sensor_readings.aggregate(avg_pipeline))
        averages = avg_res[0] if avg_res else {}
        averages.pop("_id", None)

        return jsonify({
            "total_sensor_readings":  total_readings,
            "total_rl_decisions":     total_decisions,
            "decision_counts":        decision_counts,
            "sensor_averages":        {k: round(v, 2) for k, v in averages.items()},
            "mongodb_status":         "connected",
            "database":               DB_NAME,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 7  →  Health check
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    mongo_status = "connected" if mongo_ok else "disconnected"
    if mongo_ok:
        try:
            mongo_client.admin.command("ping")
        except Exception:
            mongo_status = "error"

    return jsonify({
        "status":         "running",
        "mongodb":        mongo_status,
        "database":       DB_NAME if mongo_ok else "N/A",
        "q_table_shape":  list(q_table.shape),
        "pump_state":     pump_state,
        "endpoints": [
            "POST /predict              → RL decision + crop rec (saved to MongoDB)",
            "POST /sensor-data          → ESP32 posts readings   (saved to MongoDB)",
            "GET  /latest-reading       → Most recent sensor reading",
            "GET  /pump-state           → ESP32 polls for relay command",
            "GET  /history/sensors      → Last 100 sensor readings from MongoDB",
            "GET  /history/decisions    → Last 100 RL decisions from MongoDB",
            "GET  /history/crops        → Crop recommendation counts",
            "GET  /stats                → Summary stats from MongoDB",
            "GET  /health",
        ],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Smart Irrigation System — Flask + MongoDB Atlas")
    print("=" * 60)
    print(f"  MongoDB: {'✅ Connected to ' + DB_NAME if mongo_ok else '❌ Not connected (check .env)'}")
    print()
    print("  POST http://localhost:5000/predict")
    print("  POST http://localhost:5000/sensor-data")
    print("  GET  http://localhost:5000/latest-reading")
    print("  GET  http://localhost:5000/pump-state")
    print("  GET  http://localhost:5000/history/sensors")
    print("  GET  http://localhost:5000/history/decisions")
    print("  GET  http://localhost:5000/history/crops")
    print("  GET  http://localhost:5000/stats")
    print("  GET  http://localhost:5000/health")
    print("=" * 60 + "\n")
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT automatically
    app.run(debug=True, host="0.0.0.0", port=port)