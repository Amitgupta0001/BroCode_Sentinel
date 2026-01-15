import numpy as np
from typing import List, Dict, Tuple
from scipy import signal
from scipy import stats
import pandas as pd
from dataclasses import dataclass

@dataclass
class TemporalPattern:
    pattern_type: str
    periodicity: float
    strength: float
    consistency: float
    segments: List[Tuple[int, int]]

class TemporalAnalyzer:
    def __init__(self, window_size: int = 60, fps: int = 30):
        self.window_size = window_size
        self.fps = fps
        self.min_segment_length = 10  # Minimum frames for meaningful analysis
        
    def analyze_temporal_patterns(self, 
                                behavioral_sequence: List[float],
                                timestamps: List[float]) -> TemporalPattern:
        """Analyze temporal patterns in behavioral data"""
        if len(behavioral_sequence) < self.min_segment_length:
            return self._get_default_pattern()
        
        # Convert to numpy array
        sequence = np.array(behavioral_sequence)
        times = np.array(timestamps)
        
        # Detect periodicity
        periodicity, period_strength = self._detect_periodicity(sequence)
        
        # Analyze consistency
        consistency = self._calculate_consistency(sequence)
        
        # Segment analysis
        segments = self._segment_behavioral_sequence(sequence, times)
        
        # Classify pattern type
        pattern_type = self._classify_temporal_pattern(sequence, periodicity, consistency)
        
        return TemporalPattern(
            pattern_type=pattern_type,
            periodicity=periodicity,
            strength=period_strength,
            consistency=consistency,
            segments=segments
        )
    
    def _detect_periodicity(self, sequence: np.ndarray) -> Tuple[float, float]:
        """Detect periodicity in behavioral sequence"""
        if len(sequence) < 10:
            return 0.0, 0.0
        
        try:
            # Remove trend
            detrended = signal.detrend(sequence)
            
            # Compute autocorrelation
            autocorr = np.correlate(detrended, detrended, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation (excluding zero lag)
            peaks, properties = signal.find_peaks(autocorr[1:], height=0.1)
            
            if len(peaks) > 0:
                # First significant peak indicates period
                first_peak_lag = peaks[0] + 1  # Adjust for excluding zero lag
                periodicity = first_peak_lag / self.fps  # Convert to seconds
                
                # Strength based on peak height
                strength = properties['peak_heights'][0] / autocorr[0]
            else:
                periodicity = 0.0
                strength = 0.0
            
            return float(periodicity), float(strength)
            
        except Exception as e:
            print(f"Periodicity detection failed: {e}")
            return 0.0, 0.0
    
    def _calculate_consistency(self, sequence: np.ndarray) -> float:
        """Calculate temporal consistency of behavior"""
        if len(sequence) < 2:
            return 1.0
        
        # Calculate rolling statistics
        if len(sequence) >= 5:
            rolling_mean = pd.Series(sequence).rolling(window=5, center=True).mean().dropna()
            rolling_std = pd.Series(sequence).rolling(window=5, center=True).std().dropna()
            
            if len(rolling_std) > 0:
                # Low variability indicates high consistency
                avg_std = np.mean(rolling_std)
                max_std = np.max(rolling_std)
                consistency = 1.0 - (avg_std / max(max_std, 1e-8))
            else:
                consistency = 1.0
        else:
            # For very short sequences, use simple variance
            variance = np.var(sequence)
            consistency = 1.0 - min(variance, 1.0)
        
        return float(np.clip(consistency, 0, 1))
    
    def _segment_behavioral_sequence(self, 
                                   sequence: np.ndarray, 
                                   timestamps: np.ndarray) -> List[Tuple[int, int]]:
        """Segment behavioral sequence into meaningful chunks"""
        segments = []
        
        if len(sequence) < 2:
            return segments
        
        # Use change point detection
        change_points = self._detect_change_points(sequence)
        
        # Create segments between change points
        start_idx = 0
        for change_point in change_points:
            if change_point - start_idx >= self.min_segment_length:
                segments.append((start_idx, change_point))
            start_idx = change_point
        
        # Add final segment
        if len(sequence) - start_idx >= self.min_segment_length:
            segments.append((start_idx, len(sequence)))
        
        return segments
    
    def _detect_change_points(self, sequence: np.ndarray) -> List[int]:
        """Detect significant change points in behavioral sequence"""
        change_points = []
        
        if len(sequence) < 10:
            return change_points
        
        # Simple statistical change detection
        window_size = min(10, len(sequence) // 3)
        
        for i in range(window_size, len(sequence) - window_size):
            left_segment = sequence[i-window_size:i]
            right_segment = sequence[i:i+window_size]
            
            # Compare statistics of adjacent windows
            left_mean = np.mean(left_segment)
            right_mean = np.mean(right_segment)
            left_std = np.std(left_segment)
            right_std = np.std(right_segment)
            
            # Significance test (simplified)
            mean_diff = abs(left_mean - right_mean)
            std_pool = np.sqrt((left_std**2 + right_std**2) / 2)
            
            if std_pool > 0 and mean_diff > 2 * std_pool:
                change_points.append(i)
        
        return change_points
    
    def _classify_temporal_pattern(self, 
                                 sequence: np.ndarray,
                                 periodicity: float,
                                 consistency: float) -> str:
        """Classify the type of temporal pattern"""
        if periodicity > 0 and consistency > 0.7:
            return "rhythmic"
        elif consistency > 0.8:
            return "stable"
        elif consistency < 0.3:
            return "erratic"
        
        # Check for trend
        if len(sequence) >= 10:
            x = np.arange(len(sequence))
            slope, _, _, _, _ = stats.linregress(x, sequence)
            if abs(slope) > 0.01:
                return "trending"
        
        return "variable"
    
    def calculate_entropy_rate(self, sequence: List[float], bin_method: str = 'sturges') -> float:
        """Calculate approximate entropy rate of the time series"""
        if len(sequence) < 10:
            return 0.0
        
        # Convert to symbolic sequence (binning)
        if bin_method == 'sturges':
            n_bins = int(np.ceil(np.log2(len(sequence))) + 1
        else:
            n_bins = 10
        
        hist, bin_edges = np.histogram(sequence, bins=n_bins)
        probabilities = hist / len(sequence)
        
        # Calculate entropy
        entropy = stats.entropy(probabilities[probabilities > 0])
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(n_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    def cross_correlation_analysis(self, 
                                 sequence1: List[float],
                                 sequence2: List[float],
                                 max_lag: int = 10) -> Dict[str, float]:
        """Analyze cross-correlation between two behavioral sequences"""
        if len(sequence1) != len(sequence2) or len(sequence1) < max_lag:
            return {'max_correlation': 0.0, 'optimal_lag': 0.0, 'synchrony_score': 0.0}
        
        # Calculate cross-correlation
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(sequence1[:lag], sequence2[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(sequence1[lag:], sequence2[:-lag])[0, 1]
            else:
                corr = np.corrcoef(sequence1, sequence2)[0, 1]
            
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        correlations = np.array(correlations)
        max_corr = np.max(correlations)
        optimal_lag = np.argmax(correlations) - max_lag
        
        # Calculate synchrony score
        zero_lag_corr = correlations[max_lag]  # Correlation at lag 0
        synchrony_score = (zero_lag_corr + 1) / 2  # Convert to 0-1 scale
        
        return {
            'max_correlation': float(max_corr),
            'optimal_lag': float(optimal_lag),
            'synchrony_score': float(synchrony_score)
        }
    
    def _get_default_pattern(self) -> TemporalPattern:
        """Return default pattern for insufficient data"""
        return TemporalPattern(
            pattern_type="insufficient_data",
            periodicity=0.0,
            strength=0.0,
            consistency=0.0,
            segments=[]
        )
