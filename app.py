# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session
import json
import os
import logging
import base64
import cv2
import numpy as np
import time
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf

# Local imports (using the new modular sentinel package)
from sentinel.biometrics.keystroke_base import BilingualAuthenticationSystem
from sentinel.core.auth_engine import BehavioralAuthSystem
from sentinel.integrations.notifications import NotificationService
from sentinel.intelligence.behavioral_learning import BehavioralPatternLearner
from sentinel.integrations.api_v1 import APIKeyManager, api_bp

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("brocode_app")

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Secret management
secret = os.environ.get("BROCODE_SECRET")
if not secret:
    logger.warning("BROCODE_SECRET not set; using dev fallback. Set BROCODE_SECRET env var for production.")
    secret = "brocode_keystroke_secret_dev"
app.config['SECRET_KEY'] = secret

# CSRF protection
csrf = CSRFProtect(app)

# expose a template function to include CSRF token easily
@app.context_processor
def inject_csrf():
    return dict(csrf_token=generate_csrf)

# Models / systems
MODEL_DIR = os.environ.get("BROCODE_MODEL_DIR", "data/models")
bilingual_system = BilingualAuthenticationSystem(model_dir=MODEL_DIR)

# Behavioral orchestrator for continuous monitoring
behavioral_system = BehavioralAuthSystem(model_dir=MODEL_DIR)

# Notification service
notification_service = NotificationService()

# Behavioral pattern learner
behavioral_learner = BehavioralPatternLearner(storage_dir=MODEL_DIR)

# API Key Manager
api_key_manager = APIKeyManager(storage_dir=MODEL_DIR)
app.config['API_KEY_MANAGER'] = api_key_manager

# Register API Blueprint
app.register_blueprint(api_bp)

logger.info("BroCode Sentinel initialized with API support")

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

# Device fingerprint storage (in production, use a database)
DEVICE_DB_FILE = os.path.join(MODEL_DIR, "device_fingerprints.json")

def load_device_db():
    if os.path.exists(DEVICE_DB_FILE):
        try:
            with open(DEVICE_DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_device_db(db):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(DEVICE_DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)

device_db = load_device_db()

# Register route (keeps your original flow)
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    # POST
    username = request.form.get("username", "").strip()
    language = request.form.get("language", "english")
    keystrokes_raw = request.form.get("keystrokes", "")
    passphrase = request.form.get("passphrase", "")
    device_fingerprint_raw = request.form.get("device_fingerprint", "")
    
    if not username or len(passphrase) < 3:
        flash("Invalid username or passphrase", "error")
        return redirect(url_for("register"))

    try:
        keystrokes = json.loads(keystrokes_raw) if keystrokes_raw else []
    except Exception:
        keystrokes = []

    # Parse device fingerprint
    try:
        device_fp = json.loads(device_fingerprint_raw) if device_fingerprint_raw else {}
        device_hash = device_fp.get("hash", "unknown")
    except Exception:
        device_hash = "unknown"

    # For registration we accept multiple repetitions; here we store single sample as list-of-events
    samples = [keystrokes]
    labels = [username]

    ok = bilingual_system.register_user(username, language, samples, labels)
    if ok:
        # Tier-1 Security: Generate Blockchain DID and Verifiable Credentials
        behavioral_system.enroll_user(username)

        # Store device fingerprint for this user
        if username not in device_db:
            device_db[username] = {"trusted_devices": [], "created_at": time.time()}
        
        device_db[username]["trusted_devices"].append({
            "hash": device_hash,
            "registered_at": time.time(),
            "last_seen": time.time(),
            "device_name": f"Device {len(device_db[username]['trusted_devices']) + 1}"
        })
        save_device_db(device_db)
        
        flash(f"Registration succeeded. Device registered.", "success")
        logger.info(f"User {username} registered with device {device_hash[:16]}...")
    else:
        flash("Registration failed", "error")
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()
    flash("Session terminated.", "info")
    return redirect(url_for("index"))

# Authenticate route (keystroke-based login)
@app.route("/authenticate", methods=["GET", "POST"])
def authenticate():
    if request.method == "GET":
        return render_template("authenticate.html")
    username = request.form.get("username", "").strip()
    language = request.form.get("language", "english")
    keystrokes_raw = request.form.get("keystrokes", "")
    device_fingerprint_raw = request.form.get("device_fingerprint", "")
    geolocation_raw = request.form.get("geolocation_data", "")
    
    try:
        keystrokes = json.loads(keystrokes_raw) if keystrokes_raw else []
    except Exception:
        keystrokes = []

    # Parse device fingerprint
    try:
        device_fp = json.loads(device_fingerprint_raw) if device_fingerprint_raw else {}
        device_hash = device_fp.get("hash", "unknown")
    except Exception:
        device_hash = "unknown"

    # Parse geolocation
    try:
        geo_data = json.loads(geolocation_raw) if geolocation_raw else {}
    except Exception:
        geo_data = {}

    # Check device fingerprint
    is_known_device = False
    device_warning = ""
    location_warning = ""
    
    if username in device_db:
        trusted_devices = device_db[username].get("trusted_devices", [])
        for device in trusted_devices:
            if device["hash"] == device_hash:
                is_known_device = True
                device["last_seen"] = time.time()
                save_device_db(device_db)
                break
        
        if not is_known_device and device_hash != "unknown":
            device_warning = "⚠️ New device detected! "
            logger.warning(f"Unknown device attempting to authenticate as {username}: {device_hash[:16]}...")

        # Check for impossible travel
        location_history = device_db[username].get("location_history", [])
        if location_history and geo_data.get("ip"):
            last_location = location_history[-1]
            current_ip_data = geo_data.get("ip", {})
            
            if not current_ip_data.get("error") and last_location.get("ip"):
                last_lat = last_location["ip"].get("latitude")
                last_lon = last_location["ip"].get("longitude")
                curr_lat = current_ip_data.get("latitude")
                curr_lon = current_ip_data.get("longitude")
                last_time = last_location.get("timestamp", 0)
                curr_time = geo_data.get("timestamp", time.time() * 1000)
                
                if all([last_lat, last_lon, curr_lat, curr_lon]):
                    # Calculate distance and time
                    from math import radians, sin, cos, sqrt, atan2
                    
                    R = 6371  # Earth radius in km
                    lat1, lon1 = radians(last_lat), radians(last_lon)
                    lat2, lon2 = radians(curr_lat), radians(curr_lon)
                    
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    distance_km = R * c
                    
                    time_diff_hours = (curr_time - last_time) / (1000 * 60 * 60)
                    
                    if time_diff_hours > 0:
                        required_speed = distance_km / time_diff_hours
                        max_speed = 900  # km/h (commercial flight)
                        
                        if required_speed > max_speed:
                            location_warning = f"🚨 Impossible travel detected! ({int(distance_km)}km in {time_diff_hours:.1f}h) "
                            logger.error(f"Impossible travel for {username}: {distance_km:.0f}km in {time_diff_hours:.1f}h (requires {required_speed:.0f}km/h)")

    accepted, score = bilingual_system.authenticate_user(username, language, keystrokes)
    if accepted:
        session["user_id"] = username
        session["logged_in"] = True
        session["device_hash"] = device_hash
        session["is_known_device"] = is_known_device
        
        # Store location history
        if username in device_db:
            if "location_history" not in device_db[username]:
                device_db[username]["location_history"] = []
            
            device_db[username]["location_history"].append({
                "browser": geo_data.get("browser", {}),
                "ip": geo_data.get("ip", {}),
                "timestamp": geo_data.get("timestamp", time.time() * 1000),
                "device_hash": device_hash
            })
            
            # Keep only last 50 locations
            if len(device_db[username]["location_history"]) > 50:
                device_db[username]["location_history"] = device_db[username]["location_history"][-50:]
        
        # If unknown device, add it but flag it
        if not is_known_device and username in device_db:
            new_device = {
                "hash": device_hash,
                "registered_at": time.time(),
                "last_seen": time.time(),
                "device_name": f"New Device {len(device_db[username]['trusted_devices']) + 1}",
                "flagged": True
            }
            device_db[username]["trusted_devices"].append(new_device)
            
            # Send new device notification
            try:
                user_email = device_db[username].get("email")  # Would be set during registration
                notification_service.send_new_device_alert(username, new_device, user_email)
            except Exception as e:
                logger.error(f"Failed to send new device notification: {e}")
        
        # Send impossible travel notification if detected
        if location_warning:
            try:
                user_email = device_db[username].get("email") if username in device_db else None
                
                # Extract location info for notification
                current_ip_data = geo_data.get("ip", {})
                location_history = device_db[username].get("location_history", [])
                if location_history:
                    last_location = location_history[-2] if len(location_history) > 1 else location_history[-1]
                    
                    travel_info = {
                        "from_location": f"{last_location.get('ip', {}).get('city', 'Unknown')}, {last_location.get('ip', {}).get('country', 'Unknown')}",
                        "to_location": f"{current_ip_data.get('city', 'Unknown')}, {current_ip_data.get('country', 'Unknown')}",
                        "distance_km": int(location_warning.split("(")[1].split("km")[0]) if "km" in location_warning else 0,
                        "time_hours": float(location_warning.split("in ")[1].split("h")[0]) if "h" in location_warning else 0
                    }
                    
                    notification_service.send_impossible_travel_alert(username, travel_info, user_email)
            except Exception as e:
                logger.error(f"Failed to send impossible travel notification: {e}")
        
        save_device_db(device_db)
        
        # Combine warnings
        all_warnings = device_warning + location_warning
        flash_type = "error" if location_warning else ("warning" if device_warning else "success")
        
        flash(f"{all_warnings}Authenticated (score={score:.2f})", flash_type)
        return redirect(url_for("dashboard"))
    else:
        flash(f"Authentication failed (score={score:.2f})", "error")
        return redirect(url_for("index"))

# Dashboard route (visualization)
@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        flash("Please authenticate specifically for this session.", "error")
        return redirect(url_for("authenticate"))
    return render_template("dashboard.html")

# Notification history endpoint
@app.route("/notifications")
def notifications():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    
    username = session.get("user_id", "guest")
    history = notification_service.get_notification_history(username, limit=20)
    stats = notification_service.get_stats()
    
    return jsonify({
        "notifications": history,
        "stats": stats
    })

# Liveness check endpoint
@app.route("/liveness_check", methods=["POST"])
def liveness_check():
    """
    Verify liveness challenge response.
    Receives challenge results from frontend and validates.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400
    
    challenge_type = data.get("challenge_type")
    completed = data.get("completed", False)
    confidence = data.get("confidence", 0)
    
    # Validate liveness
    is_live = completed and confidence > 0.6
    
    # Store liveness result in session
    if session.get("logged_in"):
        session["liveness_verified"] = is_live
        session["liveness_timestamp"] = time.time()
        session["liveness_confidence"] = confidence
    
    return jsonify({
        "is_live": is_live,
        "confidence": confidence,
        "message": "Liveness verified" if is_live else "Liveness check failed",
        "challenge_type": challenge_type
    })

# Voice profile storage
VOICE_PROFILES_FILE = os.path.join(MODEL_DIR, "voice_profiles.json")

def load_voice_profiles():
    if os.path.exists(VOICE_PROFILES_FILE):
        try:
            with open(VOICE_PROFILES_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_voice_profiles(profiles):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(VOICE_PROFILES_FILE, 'w') as f:
        json.dump(profiles, f, indent=2)

voice_profiles = load_voice_profiles()

# Voice registration endpoint
@app.route("/register_voice", methods=["POST"])
def register_voice():
    """Store user's voice profile (MFCC features)"""
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400
    
    username = session.get("user_id")
    features = data.get("features")
    
    if not features:
        return jsonify({"error": "no features provided"}), 400
    
    # Store voice profile
    if username not in voice_profiles:
        voice_profiles[username] = {"samples": []}
    
    voice_profiles[username]["samples"].append({
        "features": features,
        "timestamp": time.time()
    })
    
    # Keep only last 5 samples
    if len(voice_profiles[username]["samples"]) > 5:
        voice_profiles[username]["samples"] = voice_profiles[username]["samples"][-5:]
    
    save_voice_profiles(voice_profiles)
    
    logger.info(f"Voice profile registered for {username}")
    
    return jsonify({
        "success": True,
        "message": "Voice profile registered",
        "samples_count": len(voice_profiles[username]["samples"])
    })

# Voice verification endpoint
@app.route("/verify_voice", methods=["POST"])
def verify_voice():
    """Verify user's voice against stored profile"""
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400
    
    username = session.get("user_id")
    features = data.get("features")
    
    if not features or username not in voice_profiles:
        return jsonify({
            "verified": False,
            "similarity": 0,
            "message": "No voice profile found"
        })
    
    # Compute similarity with stored samples
    stored_samples = voice_profiles[username]["samples"]
    similarities = []
    
    for sample in stored_samples:
        similarity = compute_voice_similarity(features, sample["features"])
        similarities.append(similarity)
    
    # Use average similarity
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    verified = avg_similarity > 0.7  # 70% threshold
    
    logger.info(f"Voice verification for {username}: similarity={avg_similarity:.2f}, verified={verified}")
    
    return jsonify({
        "verified": verified,
        "similarity": round(avg_similarity, 3),
        "message": "Voice verified" if verified else "Voice verification failed"
    })

def compute_voice_similarity(features1, features2):
    """Compute cosine similarity between two MFCC feature sets"""
    try:
        mean1 = features1.get("mean", [])
        mean2 = features2.get("mean", [])
        
        if not mean1 or not mean2 or len(mean1) != len(mean2):
            return 0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(mean1, mean2))
        norm1 = sum(a * a for a in mean1) ** 0.5
        norm2 = sum(b * b for b in mean2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Normalize to 0-1 range
        return (similarity + 1) / 2
    except Exception as e:
        logger.error(f"Voice similarity computation error: {e}")
        return 0

# Serve models or data for debug (optional)
@app.route("/models/<path:filename>")
def models(filename):
    return send_from_directory(MODEL_DIR, filename)

# --- New: monitor_activity endpoint for continuous monitoring ---
@app.route("/monitor_activity", methods=["POST"])
def monitor_activity():
    """
    Receives continuous data (keystrokes, frame info, device info) from frontend,
    and returns trust score + risk label.
    Algorithm checks fusion of: Keystrokes + Video.
    """
    # Enforce session login
    if not session.get("logged_in"):
         return jsonify({"error": "unauthorized", "redirect": url_for("index")}), 401

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400

    user_id = session.get("user_id", "guest") # Trust the session, not the client payload
    keystrokes = data.get("keystrokes", [])
    frame_data = data.get("frame_data", {})  
    device_info = data.get("device_info", {})
    image_b64 = data.get("image_data", None)

    # Decode image if present
    frame_arr = None
    if image_b64:
        try:
            # Remove header if present (e.g., "data:image/jpeg;base64,")
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            image_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame_arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")

    # 1) Keystroke score (0..1)
    ks_score = 0.5 # Default neutral
    if keystrokes:
        try:
            accepted, ks_score = bilingual_system.authenticate_user(user_id, device_info.get("language", "english"), keystrokes)
        except Exception:
            ks_score = 0.5 

    # Build behavioral_data input for the main behavioral_system.monitor_session
    try:
        # Pass the real frame_arr if we have it
        trust_score, anomaly_flag = behavioral_system.monitor_session(user_id, frame_data, keystrokes, frame_arr=frame_arr)
    except Exception as e:
        logger.error(f"monitor_session error: {e}")
        # Fallback
        trust_score = 0.5
        anomaly_flag = False

    # PROGRESSIVE RE-AUTHENTICATION LOGIC:
    # - Trust 0.5-1.0: Normal operation
    # - Trust 0.3-0.5: Warning state (prompt for re-auth)
    # - Trust < 0.3: Critical state (force logout)
    
    final_trust = trust_score
    if keystrokes:
        final_trust = round(0.4 * ks_score + 0.6 * trust_score, 3)

    # Determine authentication state
    auth_state = "authenticated"
    requires_reauth = False
    
    if final_trust < 0.3:
        # CRITICAL: Force logout immediately
        auth_state = "critical"
        session.clear()
        return jsonify({
            "user_id": user_id,
            "trust_score": final_trust,
            "risk": "critical",
            "auth_state": "critical",
            "message": "Session terminated due to low trust score",
            "redirect": url_for("index")
        })
    elif final_trust < 0.5:
        # WARNING: Prompt for re-authentication
        auth_state = "warning"
        requires_reauth = True
        session["reauth_required"] = True
        session["reauth_timestamp"] = time.time()
        
        # Send trust drop notification
        try:
            if username in device_db:
                user_email = device_db[username].get("email")
                notification_service.send_trust_drop_alert(user_id, final_trust, user_email)
        except Exception as e:
            logger.error(f"Failed to send trust drop notification: {e}")

    # Respond with JSON
    return jsonify({
        "user_id": user_id,
        "trust_score": final_trust,
        "risk": "high" if (anomaly_flag is True or final_trust < 0.5) else "normal",
        "anomaly": bool(anomaly_flag),
        "auth_state": auth_state,
        "requires_reauth": requires_reauth,
        "reauth_deadline": session.get("reauth_timestamp", 0) + 30 if requires_reauth else None
    })

# Run
if __name__ == "__main__":
    # If you want to pre-start any sessions or models, do that here.
    logger.info("Starting BroCode ML app (development mode)")
    app.run(debug=True)
