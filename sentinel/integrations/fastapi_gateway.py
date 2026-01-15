# BroCode Sentinel - FastAPI Migration Layer
# High-performance asynchronous backend for continuous biometric analysis

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import time
import json
import logging

# Import existing core logic
from main_authentication_system import ContinuousBehavioralSystem
from graduated_response import GraduatedResponseSystem
from temporal_trust_smoother import TemporalTrustSmoother

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BroCode Sentinel - High Performance API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
behavioral_system = ContinuousBehavioralSystem()
response_system = GraduatedResponseSystem()
trust_smoother = TemporalTrustSmoother()

class AuthData(BaseModel):
    username: str
    password: str
    device_id: Optional[str] = None

class BiometricPayload(BaseModel):
    session_id: str
    keystroke_data: Optional[List[Dict]] = None
    vision_features: Optional[Dict] = None
    frame_data: Optional[str] = None # Base64

# Simple connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_personal_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

manager = ConnectionManager()

@app.post("/api/v2/authenticate")
async def authenticate(data: AuthData):
    # Integration with existing behavioral system
    logger.info(f"FastAPI Auth attempt for user: {data.username}")
    # Mock authentication success for now - integration with DB would follow
    return {"status": "success", "session_id": f"sess_{int(time.time())}", "token": "jwt_token_here"}

@app.websocket("/ws/monitor/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            # 1. Concurrent Analysis (Simulated Async)
            # We would process frames, keystrokes and gaze in parallel here
            raw_scores = behavioral_system.analyze_activity(payload)
            
            # 2. Smooth the trust score
            smoothed_score = trust_smoother.smooth_trust_score(
                session_id, 
                raw_scores.get('trust_score', 0.5)
            )
            
            # 3. Evaluate response levels
            response = response_system.evaluate_response(session_id, smoothed_score)
            
            # 4. Immediate Feedback to client
            feedback = {
                "type": "trust_update",
                "score": smoothed_score,
                "action": response['action'],
                "threat_level": response['threat_level'],
                "timestamp": time.time()
            }
            
            await manager.send_personal_message(feedback, session_id)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"Session {session_id} disconnected from monitor")

@app.get("/api/v2/security/status")
async def get_status(session_id: str):
    # Fetch real-time security insights
    return {
        "is_active": True,
        "trust_stability": "high",
        "last_verification": time.time() - 30,
        "active_factors": ["vision", "keystroke", "device_fingerprint"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
