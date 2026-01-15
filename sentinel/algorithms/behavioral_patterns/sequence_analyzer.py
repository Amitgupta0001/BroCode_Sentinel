import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import pandas as pd

@dataclass
class BehavioralSequence:
    sequence: List[str]
    duration: float
    frequency: float
    transitions: Dict[Tuple[str, str], int]
    pattern_type: str

class SequenceAnalyzer:
    def __init__(self, min_sequence_length: int = 3, max_sequence_gap: int = 5):
        self.min_sequence_length = min_sequence_length
        self.max_sequence_gap = max_sequence_gap
        self.behavior_history = deque(maxlen=1000)
        self.sequence_patterns = defaultdict(list)
        
    def analyze_behavioral_sequences(self, behaviors: List[str], 
                                   timestamps: List[float]) -> List[BehavioralSequence]:
        """Analyze sequences of behaviors over time"""
        if len(behaviors) < self.min_sequence_length:
            return []
        
        # Update behavior history
        for behavior, timestamp in zip(behaviors, timestamps):
            self.behavior_history.append((behavior, timestamp))
        
        # Extract meaningful sequences
        sequences = self._extract_sequences(behaviors, timestamps)
        
        # Analyze each sequence
        analyzed_sequences = []
        for seq_behaviors, seq_timestamps in sequences:
            if len(seq_behaviors) >= self.min_sequence_length:
                sequence_analysis = self._analyze_single_sequence(seq_behaviors, seq_timestamps)
                analyzed_sequences.append(sequence_analysis)
        
        return analyzed_sequences
    
    def _extract_sequences(self, behaviors: List[str], timestamps: List[float]) -> List[Tuple[List[str], List[float]]]:
        """Extract continuous behavioral sequences"""
        sequences = []
        current_sequence = []
        current_timestamps = []
        
        for i, (behavior, timestamp) in enumerate(zip(behaviors, timestamps)):
            if not current_sequence:
                # Start new sequence
                current_sequence.append(behavior)
                current_timestamps.append(timestamp)
            else:
                # Check if gap is too large
                time_gap = timestamp - current_timestamps[-1]
                if time_gap <= self.max_sequence_gap:
                    current_sequence.append(behavior)
                    current_timestamps.append(timestamp)
                else:
                    # Save current sequence and start new one
                    if len(current_sequence) >= self.min_sequence_length:
                        sequences.append((current_sequence.copy(), current_timestamps.copy()))
                    current_sequence = [behavior]
                    current_timestamps = [timestamp]
        
        # Don't forget the last sequence
        if len(current_sequence) >= self.min_sequence_length:
            sequences.append((current_sequence, current_timestamps))
        
        return sequences
    
    def _analyze_single_sequence(self, behaviors: List[str], timestamps: List[float]) -> BehavioralSequence:
        """Analyze a single behavioral sequence"""
        # Calculate transitions
        transitions = defaultdict(int)
        for i in range(len(behaviors) - 1):
            transition = (behaviors[i], behaviors[i+1])
            transitions[transition] += 1
        
        # Calculate sequence metrics
        duration = timestamps[-1] - timestamps[0]
        frequency = len(behaviors) / duration if duration > 0 else 0
        
        # Classify pattern type
        pattern_type = self._classify_sequence_pattern(behaviors, transitions)
        
        return BehavioralSequence(
            sequence=behaviors,
            duration=duration,
            frequency=frequency,
            transitions=dict(transitions),
            pattern_type=pattern_type
        )
    
    def _classify_sequence_pattern(self, behaviors: List[str], 
                                 transitions: Dict[Tuple[str, str], int]) -> str:
        """Classify the type of behavioral sequence pattern"""
        unique_behaviors = set(behaviors)
        
        if len(unique_behaviors) == 1:
            return "repetitive"
        
        # Check for cyclic patterns
        if self._is_cyclic(behaviors):
            return "cyclic"
        
        # Check for progressive patterns
        if self._is_progressive(behaviors):
            return "progressive"
        
        # Check for random patterns
        if self._is_random(transitions, len(behaviors)):
            return "random"
        
        return "complex"
    
    def _is_cyclic(self, behaviors: List[str]) -> bool:
        """Check if sequence shows cyclic pattern"""
        if len(behaviors) < 6:
            return False
        
        # Look for repeating subsequences
        for cycle_length in range(2, len(behaviors) // 2):
            is_cyclic = True
            for i in range(cycle_length, len(behaviors)):
                if behaviors[i] != behaviors[i % cycle_length]:
                    is_cyclic = False
                    break
            if is_cyclic:
                return True
        
        return False
    
    def _is_progressive(self, behaviors: List[str]) -> bool:
        """Check if sequence shows progressive pattern (no backtracking)"""
        # This would typically use domain knowledge of behavior progression
        # For now, use simple heuristic: few backward transitions
        backward_transitions = 0
        behavior_indices = {behavior: i for i, behavior in enumerate(set(behaviors))}
        
        for i in range(len(behaviors) - 1):
            current_idx = behavior_indices[behaviors[i]]
            next_idx = behavior_indices[behaviors[i+1]]
            
            if next_idx < current_idx:
                backward_transitions += 1
        
        # Allow some backward transitions (up to 20%)
        max_backward = len(behaviors) * 0.2
        return backward_transitions <= max_backward
    
    def _is_random(self, transitions: Dict[Tuple[str, str], int], sequence_length: int) -> bool:
        """Check if sequence appears random"""
        if len(transitions) < 3:
            return False
        
        # Calculate transition entropy
        total_transitions = sum(transitions.values())
        entropy = 0.0
        
        for count in transitions.values():
            probability = count / total_transitions
            entropy -= probability * np.log2(probability)
        
        # High entropy suggests randomness
        max_entropy = np.log2(len(transitions))
        randomness_ratio = entropy / max_entropy if max_entropy > 0 else 0
        
        return randomness_ratio > 0.8
    
    def find_frequent_patterns(self, min_support: float = 0.1) -> List[Tuple[List[str], float]]:
        """Find frequently occurring behavioral patterns"""
        if not self.behavior_history:
            return []
        
        # Extract all sequences from history
        all_behaviors = [item[0] for item in self.behavior_history]
        all_timestamps = [item[1] for item in self.behavior_history]
        
        sequences = self._extract_sequences(all_behaviors, all_timestamps)
        
        # Count pattern occurrences
        pattern_counts = defaultdict(int)
        total_sequences = len(sequences)
        
        for seq_behaviors, _ in sequences:
            # Consider subsequences of different lengths
            for length in range(self.min_sequence_length, len(seq_behaviors) + 1):
                for start in range(len(seq_behaviors) - length + 1):
                    subsequence = tuple(seq_behaviors[start:start + length])
                    pattern_counts[subsequence] += 1
        
        # Filter by minimum support
        frequent_patterns = []
        for pattern, count in pattern_counts.items():
            support = count / total_sequences
            if support >= min_support:
                frequent_patterns.append((list(pattern), support))
        
        # Sort by support (descending)
        frequent_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return frequent_patterns
