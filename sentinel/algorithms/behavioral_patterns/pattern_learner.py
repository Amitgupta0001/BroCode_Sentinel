import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

@dataclass
class BehavioralPattern:
    pattern_id: int
    behaviors: List[str]
    transition_matrix: np.ndarray
    frequency: float
    duration_range: Tuple[float, float]
    cluster_center: np.ndarray

class PatternLearner:
    def __init__(self, n_clusters: int = 5, min_pattern_length: int = 3):
        self.n_clusters = n_clusters
        self.min_pattern_length = min_pattern_length
        self.behavior_sequences = []
        self.learned_patterns = []
        self.kmeans = None
        self.is_fitted = False
        
    def learn_behavioral_patterns(self, sequences: List[List[str]], 
                                durations: List[float]) -> List[BehavioralPattern]:
        """Learn common behavioral patterns from sequences"""
        if len(sequences) < self.n_clusters:
            warnings.warn(f"Not enough sequences ({len(sequences)}) for {self.n_clusters} clusters")
            return []
        
        self.behavior_sequences = sequences
        
        # Extract features from sequences
        sequence_features = self._extract_sequence_features(sequences, durations)
        
        # Cluster sequences
        cluster_labels = self._cluster_sequences(sequence_features)
        
        # Learn patterns from each cluster
        self.learned_patterns = self._extract_patterns_from_clusters(
            sequences, durations, cluster_labels, sequence_features
        )
        
        self.is_fitted = True
        return self.learned_patterns
    
    def _extract_sequence_features(self, sequences: List[List[str]], 
                                 durations: List[float]) -> np.ndarray:
        """Extract numerical features from behavioral sequences"""
        features = []
        
        for seq, duration in zip(sequences, durations):
            seq_features = []
            
            # Basic sequence properties
            seq_features.append(len(seq))  # Sequence length
            seq_features.append(duration)  # Duration
            seq_features.append(len(seq) / duration if duration > 0 else 0)  # Behavior frequency
            
            # Behavioral diversity
            unique_behaviors = len(set(seq))
            seq_features.append(unique_behaviors)
            seq_features.append(unique_behaviors / len(seq))  # Diversity ratio
            
            # Transition patterns
            transition_features = self._extract_transition_features(seq)
            seq_features.extend(transition_features)
            
            features.append(seq_features)
        
        return np.array(features)
    
    def _extract_transition_features(self, sequence: List[str]) -> List[float]:
        """Extract features related to behavior transitions"""
        if len(sequence) < 2:
            return [0.0] * 5
        
        # Calculate transition probabilities
        transitions = defaultdict(int)
        behavior_counts = defaultdict(int)
        
        for i in range(len(sequence) - 1):
            transition = (sequence[i], sequence[i+1])
            transitions[transition] += 1
            behavior_counts[sequence[i]] += 1
        
        # Transition entropy
        total_transitions = sum(transitions.values())
        entropy = 0.0
        for count in transitions.values():
            probability = count / total_transitions
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Self-transition ratio
        self_transitions = sum(1 for (b1, b2) in transitions.keys() if b1 == b2)
        self_transition_ratio = self_transitions / total_transitions if total_transitions > 0 else 0
        
        # Unique transition ratio
        unique_transitions = len(transitions)
        max_possible_transitions = len(set(sequence)) ** 2
        transition_diversity = unique_transitions / max_possible_transitions if max_possible_transitions > 0 else 0
        
        # Most frequent transition
        if transitions:
            max_transition_count = max(transitions.values())
            dominant_transition_ratio = max_transition_count / total_transitions
        else:
            dominant_transition_ratio = 0
        
        return [
            entropy,
            self_transition_ratio,
            transition_diversity,
            dominant_transition_ratio,
            total_transitions / len(sequence)  # Average transitions per behavior
        ]
    
    def _cluster_sequences(self, features: np.ndarray) -> np.ndarray:
        """Cluster sequences using K-means"""
        # Determine optimal number of clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = self._find_optimal_clusters(features)
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(features)
        
        return cluster_labels
    
    def _find_optimal_clusters(self, features: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using silhouette score"""
        if len(features) <= 2:
            return 1
        
        best_k = 2
        best_score = -1
        
        for k in range(2, min(max_k, len(features))):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def _extract_patterns_from_clusters(self, sequences: List[List[str]],
                                      durations: List[float],
                                      cluster_labels: np.ndarray,
                                      features: np.ndarray) -> List[BehavioralPattern]:
        """Extract patterns from each cluster"""
        patterns = []
        
        for cluster_id in range(self.n_clusters):
            cluster_sequences = [seq for i, seq in enumerate(sequences) 
                               if cluster_labels[i] == cluster_id]
            cluster_durations = [dur for i, dur in enumerate(durations) 
                               if cluster_labels[i] == cluster_id]
            cluster_features = features[cluster_labels == cluster_id]
            
            if not cluster_sequences:
                continue
            
            # Calculate cluster statistics
            pattern_frequency = len(cluster_sequences) / len(sequences)
            duration_range = (min(cluster_durations), max(cluster_durations))
            
            # Find representative sequence
            if len(cluster_features) > 0:
                cluster_center = self.kmeans.cluster_centers_[cluster_id]
                # Find sequence closest to cluster center
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                representative_idx = np.argmin(distances)
                representative_sequence = cluster_sequences[representative_idx]
            else:
                representative_sequence = cluster_sequences[0]
            
            # Calculate transition matrix for representative pattern
            transition_matrix = self._build_transition_matrix(representative_sequence)
            
            pattern = BehavioralPattern(
                pattern_id=cluster_id,
                behaviors=representative_sequence,
                transition_matrix=transition_matrix,
                frequency=pattern_frequency,
                duration_range=duration_range,
                cluster_center=cluster_center
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _build_transition_matrix(self, sequence: List[str]) -> np.ndarray:
        """Build transition probability matrix for a sequence"""
        behaviors = sorted(set(sequence))
        n_behaviors = len(behaviors)
        behavior_to_idx = {behavior: i for i, behavior in enumerate(behaviors)}
        
        # Initialize transition matrix
        transition_matrix = np.zeros((n_behaviors, n_behaviors))
        
        # Count transitions
        for i in range(len(sequence) - 1):
            current_idx = behavior_to_idx[sequence[i]]
            next_idx = behavior_to_idx[sequence[i+1]]
            transition_matrix[current_idx, next_idx] += 1
        
        # Convert to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    out=np.zeros_like(transition_matrix), 
                                    where=row_sums != 0)
        
        return transition_matrix
    
    def predict_pattern(self, sequence: List[str], duration: float) -> Tuple[int, float]:
        """Predict which pattern a new sequence belongs to"""
        if not self.is_fitted:
            raise ValueError("Pattern learner must be fitted first")
        
        # Extract features for the new sequence
        features = self._extract_sequence_features([sequence], [duration])
        
        # Predict cluster
        cluster_id = self.kmeans.predict(features)[0]
        
        # Calculate confidence (distance to cluster center)
        distance = np.linalg.norm(features - self.kmeans.cluster_centers_[cluster_id])
        max_distance = np.max(np.linalg.norm(
            self.kmeans.cluster_centers_ - self.kmeans.cluster_centers_[cluster_id], 
            axis=1
        ))
        confidence = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        return cluster_id, float(confidence)
    
    def get_pattern_summary(self) -> Dict:
        """Get summary of learned patterns"""
        if not self.learned_patterns:
            return {}
        
        summary = {
            'total_patterns': len(self.learned_patterns),
            'pattern_frequencies': {},
            'average_durations': {},
            'pattern_lengths': {}
        }
        
        for pattern in self.learned_patterns:
            summary['pattern_frequencies'][pattern.pattern_id] = pattern.frequency
            summary['average_durations'][pattern.pattern_id] = np.mean(pattern.duration_range)
            summary['pattern_lengths'][pattern.pattern_id] = len(pattern.behaviors)
        
        return summary
