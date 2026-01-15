# Anti-Spoofing Module
# Implements liveness detection through blink detection and micro-movements

import cv2
import numpy as np
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class AntiSpoofingDetector:
    """
    Advanced anti-spoofing detection using:
    - Blink detection
    - Head micro-movements
    - Texture analysis
    - Screen replay detection
    """
    
    def __init__(self):
        # Eye aspect ratio (EAR) for blink detection
        self.EAR_THRESHOLD = 0.25
        self.BLINK_FRAMES = 3
        
        # Blink tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_time = 0
        self.blink_history = deque(maxlen=10)
        
        # Movement tracking
        self.movement_history = deque(maxlen=30)
        self.last_face_position = None
        
        # Texture analysis
        self.texture_scores = deque(maxlen=10)
        
        # Load face cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        except Exception as e:
            logger.error(f"Error loading cascades: {e}")
            self.face_cascade = None
            self.eye_cascade = None
    
    def detect_liveness(self, frame):
        """
        Comprehensive liveness detection
        
        Args:
            frame: Video frame (numpy array)
        
        Returns:
            Dict with liveness results
        """
        if frame is None or self.face_cascade is None:
            return {
                'is_live': False,
                'confidence': 0.0,
                'reason': 'No frame or cascade not loaded'
            }
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return {
                'is_live': False,
                'confidence': 0.0,
                'reason': 'No face detected'
            }
        
        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        
        # Run all liveness checks
        blink_result = self._detect_blink(gray, face)
        movement_result = self._detect_movement(face)
        texture_result = self._analyze_texture(face_roi)
        depth_result = self._estimate_depth(face_roi_color)
        
        # Combine results
        liveness_score = self._combine_liveness_scores(
            blink_result,
            movement_result,
            texture_result,
            depth_result
        )
        
        is_live = liveness_score > 0.6
        
        return {
            'is_live': is_live,
            'confidence': round(liveness_score, 3),
            'blink_detected': blink_result['blink_detected'],
            'total_blinks': self.total_blinks,
            'movement_score': movement_result['score'],
            'texture_score': texture_result['score'],
            'depth_score': depth_result['score'],
            'checks': {
                'blink': blink_result,
                'movement': movement_result,
                'texture': texture_result,
                'depth': depth_result
            }
        }
    
    def _detect_blink(self, gray, face):
        """Detect eye blinks using eye aspect ratio"""
        x, y, w, h = face
        
        # Extract eye region (upper half of face)
        eye_region = gray[y:y+int(h*0.6), x:x+w]
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        blink_detected = False
        ear = 0.5  # Default eye aspect ratio
        
        if len(eyes) >= 2:
            # Calculate eye aspect ratio (simplified)
            # In production, use dlib or mediapipe for accurate landmarks
            eye_heights = [e[3] for e in eyes[:2]]
            eye_widths = [e[2] for e in eyes[:2]]
            
            avg_height = np.mean(eye_heights)
            avg_width = np.mean(eye_widths)
            
            ear = avg_height / (avg_width + 1e-6)
            
            # Check if eyes are closed
            if ear < self.EAR_THRESHOLD:
                self.blink_counter += 1
            else:
                if self.blink_counter >= self.BLINK_FRAMES:
                    # Blink detected
                    current_time = time.time()
                    if current_time - self.last_blink_time > 0.3:  # Debounce
                        self.total_blinks += 1
                        self.last_blink_time = current_time
                        self.blink_history.append(current_time)
                        blink_detected = True
                
                self.blink_counter = 0
        
        # Calculate blink rate (blinks per minute)
        blink_rate = 0
        if len(self.blink_history) >= 2:
            time_span = self.blink_history[-1] - self.blink_history[0]
            if time_span > 0:
                blink_rate = (len(self.blink_history) / time_span) * 60
        
        # Normal blink rate: 15-20 per minute
        blink_score = 0.5
        if 10 <= blink_rate <= 30:
            blink_score = 1.0
        elif 5 <= blink_rate <= 40:
            blink_score = 0.7
        
        return {
            'blink_detected': blink_detected,
            'total_blinks': self.total_blinks,
            'blink_rate': round(blink_rate, 1),
            'ear': round(ear, 3),
            'score': blink_score
        }
    
    def _detect_movement(self, face):
        """Detect natural head micro-movements"""
        x, y, w, h = face
        
        # Calculate face center
        center_x = x + w // 2
        center_y = y + h // 2
        current_position = (center_x, center_y)
        
        movement_score = 0.5
        movement_detected = False
        
        if self.last_face_position is not None:
            # Calculate movement
            dx = abs(current_position[0] - self.last_face_position[0])
            dy = abs(current_position[1] - self.last_face_position[1])
            movement = np.sqrt(dx**2 + dy**2)
            
            self.movement_history.append(movement)
            
            # Analyze movement pattern
            if len(self.movement_history) >= 10:
                movements = list(self.movement_history)
                avg_movement = np.mean(movements)
                std_movement = np.std(movements)
                
                # Natural movement: some variation but not too much
                if 1 < avg_movement < 20 and std_movement > 0.5:
                    movement_score = 1.0
                    movement_detected = True
                elif avg_movement < 0.5:
                    # Too still - might be photo
                    movement_score = 0.3
                elif avg_movement > 30:
                    # Too much movement - might be video
                    movement_score = 0.4
                else:
                    movement_score = 0.7
        
        self.last_face_position = current_position
        
        return {
            'movement_detected': movement_detected,
            'score': movement_score,
            'avg_movement': round(np.mean(list(self.movement_history)), 2) if self.movement_history else 0
        }
    
    def _analyze_texture(self, face_roi):
        """Analyze texture to detect screen replay"""
        # Calculate texture complexity using Laplacian variance
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        variance = laplacian.var()
        
        self.texture_scores.append(variance)
        
        # Real faces have higher texture variance than screens
        # Typical ranges: Real face: 100-1000, Screen: 10-100
        texture_score = 0.5
        
        if variance > 100:
            texture_score = 1.0
        elif variance > 50:
            texture_score = 0.7
        elif variance < 20:
            texture_score = 0.2
        
        return {
            'variance': round(variance, 2),
            'score': texture_score,
            'avg_variance': round(np.mean(list(self.texture_scores)), 2) if self.texture_scores else 0
        }
    
    def _estimate_depth(self, face_roi_color):
        """Estimate depth using color analysis"""
        # Real faces have color variation, screens are flatter
        
        # Convert to HSV
        hsv = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2HSV)
        
        # Calculate color variance in each channel
        h_var = np.var(hsv[:, :, 0])
        s_var = np.var(hsv[:, :, 1])
        v_var = np.var(hsv[:, :, 2])
        
        # Real faces have higher saturation variance
        depth_score = 0.5
        
        if s_var > 200:
            depth_score = 1.0
        elif s_var > 100:
            depth_score = 0.7
        elif s_var < 50:
            depth_score = 0.3
        
        return {
            'h_variance': round(h_var, 2),
            's_variance': round(s_var, 2),
            'v_variance': round(v_var, 2),
            'score': depth_score
        }
    
    def _combine_liveness_scores(self, blink, movement, texture, depth):
        """Combine all liveness scores"""
        # Weighted combination
        weights = {
            'blink': 0.35,
            'movement': 0.25,
            'texture': 0.25,
            'depth': 0.15
        }
        
        combined_score = (
            blink['score'] * weights['blink'] +
            movement['score'] * weights['movement'] +
            texture['score'] * weights['texture'] +
            depth['score'] * weights['depth']
        )
        
        return combined_score
    
    def request_liveness_challenge(self):
        """
        Generate a liveness challenge
        
        Returns:
            Dict with challenge instructions
        """
        import random
        
        challenges = [
            {
                'type': 'blink',
                'instruction': 'Please blink 3 times',
                'expected_blinks': 3,
                'timeout': 5
            },
            {
                'type': 'head_turn',
                'instruction': 'Please turn your head left, then right',
                'expected_movements': 2,
                'timeout': 5
            },
            {
                'type': 'smile',
                'instruction': 'Please smile',
                'timeout': 3
            }
        ]
        
        return random.choice(challenges)
    
    def verify_challenge_response(self, challenge, response_data):
        """
        Verify liveness challenge response
        
        Args:
            challenge: Challenge dict
            response_data: Response data from user
        
        Returns:
            Boolean indicating if challenge passed
        """
        if challenge['type'] == 'blink':
            # Check if user blinked required number of times
            return response_data.get('blinks', 0) >= challenge['expected_blinks']
        
        elif challenge['type'] == 'head_turn':
            # Check if user moved head
            return response_data.get('movements', 0) >= challenge['expected_movements']
        
        elif challenge['type'] == 'smile':
            # Check if smile detected (simplified)
            return response_data.get('smile_detected', False)
        
        return False
    
    def reset(self):
        """Reset detector state"""
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_time = 0
        self.blink_history.clear()
        self.movement_history.clear()
        self.last_face_position = None
        self.texture_scores.clear()
    
    def get_statistics(self):
        """Get anti-spoofing statistics"""
        return {
            'total_blinks': self.total_blinks,
            'blink_rate': round((len(self.blink_history) / 60) * 60, 1) if self.blink_history else 0,
            'avg_movement': round(np.mean(list(self.movement_history)), 2) if self.movement_history else 0,
            'avg_texture': round(np.mean(list(self.texture_scores)), 2) if self.texture_scores else 0
        }
