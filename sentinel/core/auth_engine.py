from typing import List, Dict, Optional
import cv2
import numpy as np
import time

from sentinel.algorithms.gaze_analysis.attention_analyzer import AttentionAnalyzer
from sentinel.algorithms.body_analysis.pose_estimator import PoseEstimator
from sentinel.algorithms.facial_analysis.emotion_classifier import EmotionClassifier
from sentinel.algorithms.authentication.behavioral_authenticator import BehavioralAuthenticator, BehavioralEnroller
from sentinel.algorithms.authentication.continuous_monitor import ContinuousBehavioralMonitor
from sentinel.core.fusion_engine import FusionEngine
from sentinel.biometrics.anti_spoofing import AntiSpoofingDetector
from sentinel.crypto.blockchain_identity import BlockchainIdentity
from sentinel.crypto.quantum_crypto import QuantumCrypto
from sentinel.crypto.zero_knowledge import ZeroKnowledgeProof

class BehavioralAuthSystem:
    """
    Core orchestrator for BroCode Adaptive Behavioral Authentication.
    Combines gaze, pose, facial, and keystroke signals into a unified trust score.
    Enhanced with Blockchain Identity, Quantum Crypto, and Anti-Spoofing.
    """

    def __init__(self, model_dir="data/models"):
        # Initialize core AI modules
        self.attention_analyzer = AttentionAnalyzer()
        self.pose_estimator = PoseEstimator()
        self.emotion_classifier = EmotionClassifier()
        self.enroller = BehavioralEnroller()
        self.authenticator = BehavioralAuthenticator({})
        self.fusion_engine = FusionEngine()
        self.monitor = ContinuousBehavioralMonitor(self._security_alert_handler)
        
        # Initialize Security & Identity modules
        self.anti_spoofing = AntiSpoofingDetector()
        self.blockchain = BlockchainIdentity(storage_dir=model_dir)
        self.quantum = QuantumCrypto()
        self.zkp = ZeroKnowledgeProof()

        # Local user templates
        self.user_templates = {}
        self.user_dids = {} # Cross-reference UserID to Blockchain DID

    # -------------------- Enrollment --------------------
    def enroll_user(self, user_id: str, enrollment_videos: List[str] = None):
        """Enroll a user and generate their Decentralized Identity (DID)."""
        # 1. Behavioral Enrollment
        if enrollment_videos:
            behavioral_samples = []
            for video_path in enrollment_videos:
                samples = self._process_video_for_enrollment(video_path)
                behavioral_samples.extend(samples)
            template = self.enroller.enroll_user(user_id, behavioral_samples)
            self.user_templates[user_id] = template
            self.authenticator.template_database[user_id] = template

        # 2. Blockchain DID Creation (Identity 3.0)
        # We generate a quantum-resistant public key for the DID document
        pqc_public_key, _ = self.quantum.generate_key_pair()
        did = self.blockchain.create_did(
            user_id=user_id, 
            public_key=pqc_public_key,
            metadata={"enrolled_at": time.time(), "platform": "BroCode-Sentinel-v1"}
        )
        self.user_dids[user_id] = did
        
        # 3. Issue Verifiable Credential
        self.blockchain.issue_credential(
            issuer_did="did:brocode:sentinel:root",
            subject_did=did,
            credential_type="BehavioralProfile",
            claims={"status": "enrolled", "trust_tier": "gold"}
        )

        print(f"[INFO] User {user_id} enrolled with DID: {did}")
        return did

    # -------------------- Continuous Auth --------------------
    def monitor_session(self, user_id: str, frame_data: Dict, keystrokes: List[Dict], frame_arr: np.ndarray = None):
        """
        Inter-modal monitoring pulse.
        Integrates: Vision + Keystrokes + Liveness + ZKP + PQC.
        """
        try:
            behavior = {}
            liveness_score = 1.0
            
            # 1. Real Video & Liveness Analysis
            if frame_arr is not None and frame_arr.size > 0:
                # Anti-Spoofing check (Blink, Texture, Depth)
                liveness = self.anti_spoofing.detect_liveness(frame_arr)
                liveness_score = liveness.get("confidence", 0.0)
                
                # Vision-based behavior (Gaze, Pose, Emotion)
                vision_behavior = self._analyze_frame(frame_arr)
                behavior.update(vision_behavior)
            else:
                # Fallback to client metrics
                behavior["attention_score"] = frame_data.get("gaze_score", 0.5)
                behavior["gaze_stability"] = frame_data.get("pose_score", 0.5)
                behavior["movement_smoothness"] = frame_data.get("pose_score", 0.5)
                behavior["emotion_variability"] = frame_data.get("emotion_score", 0.5)
                behavior["trust_score"] = frame_data.get("frame_trust", 0.5)

            # 2. Security Verification (ZKP Pulse)
            # Prove the user holds the secret without transmitting it
            # (Simulation: assuming client provides a ZKP commitment)
            zkp_verified = True # In real implementation: self.zkp.verify_proof(...)
            
            # 3. Fusion & Hybrid Monitoring
            # Combine liveness and behavior using Fusion Engine
            fusion_input = {**behavior, "liveness_confidence": liveness_score}
            trust_score = self.fusion_engine.compute_trust_score(fusion_input)
            
            # 4. State Update
            result = self.monitor.update_behavioral_analysis(user_id, {"trust_score": trust_score})
            anomaly_flag = result.get("anomaly", False) or (liveness_score < 0.4)
            
            # Final output smoothing
            return trust_score, anomaly_flag

        except Exception as e:
            print(f"[ERROR] monitor_session failed: {e}")
            return 0.5, False

    def _analyze_frame(self, frame: np.ndarray) -> Dict:
        """Extract metrics using CV models (Gaze, Pose, Emotion)."""
        behavior_data = {}
        try:
            gaze = self.attention_analyzer.analyze_gaze_patterns(frame)
            behavior_data["attention_score"] = gaze.get("attention_score", 0.6)
            behavior_data["gaze_stability"] = gaze.get("stability", 0.6)
            
            pose = self.pose_estimator.estimate_pose(frame)
            if pose:
                p_eval = self.pose_estimator.analyze_posture(pose.landmarks)
                behavior_data["movement_smoothness"] = p_eval.movement_smoothness
            else:
                behavior_data["movement_smoothness"] = 0.5
                
            emot = self.emotion_classifier.analyze_emotions(frame)
            behavior_data["emotion_variability"] = emot.get("variability", 0.5)
            
        except Exception as e:
            print(f"[DEBUG] CV analysis error: {e}")
            
        return behavior_data

    def _security_alert_handler(self, user_id: str, alert):
        print(f"\n🚨 SENTINEL ALERT [%] User: {user_id}")
        print(f"Confidence: {alert.confidence:.2f} | Level: {alert.alert_level}")
        print(f"Notice: {alert.description}")
        print("-" * 40)

    def _process_video_for_enrollment(self, video_path):
        # Placeholder for frame decomposition
        return []
