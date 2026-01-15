# 🧪 BroCode Sentinel: Unified Testing Guide

## 🚦 System Prerequisites
- **Python 3.9+**
- **OpenCV** & **NumPy**
- **Flask** & **FastAPI** (for gateway testing)
- **Webcam access**

---

## 🛠️ Environment Setup
```bash
# Registering the sentinel package (Optional but recommended)
python -m pip install -e .

# Start the primary security engine
python app.py
```
Access at: `http://localhost:5000`

---

## 🔐 Core Authentication Scenarios

### **1. Registration & Profiling**
- **Action**: Register as a new user.
- **Requirement**: Type the security phrase **at least 5-8 times** to build a robust `keystroke_enhanced` profile.
- **Verification**: Check `data/models/user_keystroke_profiles.json` for new entries.

### **2. Continuous Trust Monitoring**
- **Action**: Stay in front of the camera and type occasionally.
- **Verification**: 
    - Dashboard chart should update every 5 seconds.
    - Status should reflect "NORMAL" security.
    - Check terminal for `sentinel.core.auth_engine` logs.

### **3. Anti-Spoofing & Spoof Testing**
- **Test A (Face Removal)**: Cover camera or leave the frame.
    - *Expected*: Trust score drops within 5-10s; Graduation to `WARN` or `CHALLENGE`.
- **Test B (Photo Spoofing)**: Hold a photo/video of yourself up to the camera.
    - *Expected*: `sentinel.biometrics.anti_spoofing` detects lack of micro-movement/texture match; Trust drops.

---

## ⚡ Advanced Feature Testing

### **1. Post-Quantum Cryptography**
- **Command**: Run `python sentinel/crypto/quantum_crypto.py` (if set as main script).
- **Manual**: Verify that `data/models/quantum_keys.json` contains Kyber-768 and Dilithium-3 keys.

### **2. Blockchain DID Registry**
- **Action**: Register/Login and check `data/models/did_registry.json`.
- **Verification**: Each user should have a unique `did:brocode:` identifier.

### **3. FastAPI WebSocket Performance**
- **Start Gateway**: `uvicorn sentinel.integrations.fastapi_gateway:app --port 8000`
- **Testing**: Connect a test WebSocket client to `ws://localhost:8000/ws/monitor/{session_id}`.

---

## 🛡️ Trust Graduation Testing

| Scenario | Raw Trust | Graduated Action | User Impact |
|:---|:---:|:---|:---|
| **Steady state** | 0.9 | `ALLOW` | Transparent |
| **Partial Occlusion** | 0.4 - 0.5 | `WARN` | UI Banner appears |
| **Anomaly Detected** | 0.3 | `CHALLENGE` | Biometric Popup |
| **No Face Detected** | < 0.2 | `LOGOUT` | Immediate redirect |

---

## 📂 Internal File Mapping (for Debugging)
- **Trust Drops?** Check `sentinel/core/trust_smoother.py`
- **Scoring Issues?** Check `sentinel/intelligence/risk_scoring.py`
- **Vision Faults?** Check `sentinel/algorithms/gaze_analysis`
- **Client Errors?** Check `static/monitor.js`

---

## ✅ Success Benchmarks
- [ ] **False Logout Frequency**: < 2% with temporal smoothing.
- [ ] **Spoof Detection Latency**: < 8 seconds.
- [ ] **Trust Accuracy**: Highly correlated with user presence.
- [ ] **Cross-Platform**: JS SDK operates in Browser; Mobile SDK registers successfully.

**Sentinel is ready for operation.** 🚀🛡️
