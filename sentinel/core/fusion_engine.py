# Adaptive Weight Fusion Module
# Dynamically adjusts fusion weights based on context

import logging
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveFusionEngine:
    """
    Intelligent fusion engine that adapts weights based on context.
    Replaces static weights with dynamic, context-aware weighting.
    """
    
    def __init__(self):
        # Base weights (fallback)
        self.base_weights = {
            'keystroke': 0.25,
            'face': 0.30,
            'behavior': 0.20,
            'liveness': 0.15,
            'voice': 0.10
        }
        
        # Confidence thresholds for each modality
        self.confidence_thresholds = {
            'keystroke': 0.5,
            'face': 0.6,
            'behavior': 0.5,
            'liveness': 0.7,
            'voice': 0.6
        }
    
    def fuse_scores(self, scores, context=None):
        """
        Adaptively fuse scores based on context
        
        Args:
            scores: Dict of modality scores {modality: score}
            context: Dict with contextual information:
                - typing_active: Boolean
                - lighting_quality: float (0-1)
                - camera_available: Boolean
                - microphone_available: Boolean
                - user_activity: str ('typing', 'reading', 'idle')
        
        Returns:
            Dict with fused score and metadata
        """
        context = context or {}
        
        # Calculate adaptive weights
        adaptive_weights = self._calculate_adaptive_weights(scores, context)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            normalized_weights = {
                k: v / total_weight 
                for k, v in adaptive_weights.items()
            }
        else:
            normalized_weights = self.base_weights.copy()
        
        # Calculate weighted fusion
        fused_score = 0.0
        contributions = {}
        
        for modality, score in scores.items():
            if modality in normalized_weights:
                weight = normalized_weights[modality]
                contribution = score * weight
                fused_score += contribution
                contributions[modality] = {
                    'score': score,
                    'weight': weight,
                    'contribution': contribution
                }
        
        # Calculate confidence in fusion
        fusion_confidence = self._calculate_fusion_confidence(
            scores, 
            normalized_weights,
            context
        )
        
        return {
            'fused_score': round(fused_score, 3),
            'adaptive_weights': {k: round(v, 3) for k, v in normalized_weights.items()},
            'base_weights': self.base_weights,
            'contributions': contributions,
            'fusion_confidence': round(fusion_confidence, 3),
            'context': context
        }
    
    def _calculate_adaptive_weights(self, scores, context):
        """Calculate context-aware weights"""
        weights = self.base_weights.copy()
        
        # Adjust based on typing activity
        typing_active = context.get('typing_active', False)
        if typing_active:
            # Increase keystroke weight when typing
            weights['keystroke'] *= 1.5
        else:
            # Decrease keystroke weight when not typing
            weights['keystroke'] *= 0.5
            # Increase vision weights
            weights['face'] *= 1.2
            weights['liveness'] *= 1.2
        
        # Adjust based on lighting quality
        lighting_quality = context.get('lighting_quality', 0.8)
        if lighting_quality < 0.5:
            # Poor lighting - reduce vision weights
            weights['face'] *= 0.6
            weights['liveness'] *= 0.6
            # Increase other weights
            weights['keystroke'] *= 1.3
            weights['behavior'] *= 1.2
        elif lighting_quality > 0.8:
            # Good lighting - increase vision weights
            weights['face'] *= 1.2
            weights['liveness'] *= 1.2
        
        # Adjust based on camera availability
        camera_available = context.get('camera_available', True)
        if not camera_available:
            # No camera - zero out vision weights
            weights['face'] = 0.0
            weights['liveness'] = 0.0
            # Boost other weights
            weights['keystroke'] *= 1.5
            weights['behavior'] *= 1.5
            weights['voice'] *= 1.3
        
        # Adjust based on microphone availability
        microphone_available = context.get('microphone_available', True)
        if not microphone_available:
            weights['voice'] = 0.0
        
        # Adjust based on user activity
        activity = context.get('user_activity', 'unknown')
        if activity == 'typing':
            weights['keystroke'] *= 1.3
        elif activity == 'reading':
            weights['face'] *= 1.2
            weights['behavior'] *= 1.2
        elif activity == 'idle':
            weights['face'] *= 1.1
            weights['liveness'] *= 1.1
        
        # Adjust based on score confidence
        for modality, score in scores.items():
            if modality in weights:
                # Reduce weight for low-confidence scores
                if score < 0.3:
                    weights[modality] *= 0.7
                # Increase weight for high-confidence scores
                elif score > 0.8:
                    weights[modality] *= 1.1
        
        return weights
    
    def _calculate_fusion_confidence(self, scores, weights, context):
        """Calculate confidence in the fused score"""
        confidences = []
        
        # Confidence based on number of active modalities
        active_modalities = sum(1 for w in weights.values() if w > 0)
        modality_confidence = min(active_modalities / len(self.base_weights), 1.0)
        confidences.append(modality_confidence)
        
        # Confidence based on score agreement
        if len(scores) >= 2:
            score_values = list(scores.values())
            score_std = np.std(score_values)
            agreement_confidence = max(0, 1.0 - score_std)
            confidences.append(agreement_confidence)
        
        # Confidence based on context quality
        context_quality = 1.0
        if not context.get('camera_available', True):
            context_quality *= 0.8
        if not context.get('microphone_available', True):
            context_quality *= 0.9
        if context.get('lighting_quality', 0.8) < 0.5:
            context_quality *= 0.7
        confidences.append(context_quality)
        
        # Overall confidence
        return np.mean(confidences)
    
    def explain_weights(self, scores, context):
        """
        Provide human-readable explanation of weight adjustments
        
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Typing activity
        if context.get('typing_active'):
            explanations.append("✅ Keystroke weight increased (active typing detected)")
        else:
            explanations.append("⬇️ Keystroke weight decreased (no typing activity)")
        
        # Lighting
        lighting = context.get('lighting_quality', 0.8)
        if lighting < 0.5:
            explanations.append("⚠️ Vision weights reduced (poor lighting)")
        elif lighting > 0.8:
            explanations.append("✅ Vision weights increased (good lighting)")
        
        # Camera
        if not context.get('camera_available', True):
            explanations.append("❌ Vision disabled (camera unavailable)")
        
        # Microphone
        if not context.get('microphone_available', True):
            explanations.append("❌ Voice disabled (microphone unavailable)")
        
        # Activity
        activity = context.get('user_activity', 'unknown')
        if activity == 'typing':
            explanations.append("⌨️ Keystroke weight boosted (typing activity)")
        elif activity == 'reading':
            explanations.append("👁️ Vision weights boosted (reading activity)")
        
        # Score-based adjustments
        for modality, score in scores.items():
            if score < 0.3:
                explanations.append(f"⬇️ {modality.capitalize()} weight reduced (low confidence)")
            elif score > 0.8:
                explanations.append(f"⬆️ {modality.capitalize()} weight increased (high confidence)")
        
        return explanations
    
    def get_optimal_modalities(self, context):
        """
        Suggest which modalities should be prioritized given context
        
        Returns:
            List of modalities sorted by priority
        """
        # Calculate weights for empty scores (just based on context)
        dummy_scores = {k: 0.5 for k in self.base_weights.keys()}
        weights = self._calculate_adaptive_weights(dummy_scores, context)
        
        # Sort by weight
        sorted_modalities = sorted(
            weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [m for m, w in sorted_modalities if w > 0]

# Alias for backward compatibility
FusionEngine = AdaptiveFusionEngine
