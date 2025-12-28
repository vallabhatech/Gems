"""
Enhanced Boundary/Transition Analysis - LAYER 3C
Focuses on scene boundaries and frame transitions where deepfakes often fail
"""
import cv2
import numpy as np


def analyze_boundaries(frame_paths, scene_boundaries, timestamps):
    """
    Enhanced analysis of scene boundaries and transitions
    
    Args:
        frame_paths: List of frame paths
        scene_boundaries: Indices of frames at scene changes
        timestamps: Frame timestamps
    
    Returns:
        dict: {
            'score': float (0-1, higher = more suspicious),
            'boundary_anomalies': int,
            'quality_drops': int,
            'suspicious_transitions': list
        }
    """
    try:
        results = {
            'score': 0.0,
            'boundary_anomalies': 0,
            'quality_drops': 0,
            'suspicious_transitions': [],
            'analyzed_boundaries': 0
        }
        
        if len(frame_paths) < 2:
            return results
        
        # Focus on scene boundaries
        boundary_indices = set(scene_boundaries) if scene_boundaries else set()
        
        # Add first/last 5 frames as boundaries
        boundary_indices.update(range(min(5, len(frame_paths))))
        boundary_indices.update(range(max(0, len(frame_paths) - 5), len(frame_paths)))
        
        boundary_indices = sorted(list(boundary_indices))
        
        # Analyze each boundary region
        for idx in boundary_indices:
            if idx >= len(frame_paths) - 1:
                continue
            
            # Compare with next frame
            current_frame = cv2.imread(frame_paths[idx])
            next_frame = cv2.imread(frame_paths[idx + 1])
            
            if current_frame is None or next_frame is None:
                continue
            
            results['analyzed_boundaries'] += 1
            
            # Check 1: Quality consistency
            quality_change = check_quality_drop(current_frame, next_frame)
            if quality_change > 0.3:  # Significant quality drop
                results['quality_drops'] += 1
                results['suspicious_transitions'].append({
                    'frame_index': idx,
                    'timestamp': timestamps[idx] if idx < len(timestamps) else 0,
                    'type': 'quality_drop',
                    'severity': float(quality_change)
                })
            
            # Check 2: Color/lighting consistency
            color_shift = check_color_shift(current_frame, next_frame)
            if color_shift > 0.35:  # Sudden color change
                results['boundary_anomalies'] += 1
                results['suspicious_transitions'].append({
                    'frame_index': idx,
                    'timestamp': timestamps[idx] if idx < len(timestamps) else 0,
                    'type': 'color_shift',
                    'severity': float(color_shift)
                })
            
            # Check 3: Structural similarity drop
            structural_change = check_structural_change(current_frame, next_frame)
            if structural_change > 0.4:  # Large structural change (not normal motion)
                results['boundary_anomalies'] += 1
                results['suspicious_transitions'].append({
                    'frame_index': idx,
                    'timestamp': timestamps[idx] if idx < len(timestamps) else 0,
                    'type': 'structural_anomaly',
                    'severity': float(structural_change)
                })
        
        # Calculate score
        if results['analyzed_boundaries'] > 0:
            anomaly_rate = (results['quality_drops'] + results['boundary_anomalies']) / results['analyzed_boundaries']
            results['score'] = min(anomaly_rate * 2.0, 1.0)
        
        return results
        
    except Exception as e:
        print(f"Boundary analysis error: {e}")
        return {
            'score': 0.0,
            'error': str(e)
        }


def check_quality_drop(frame1, frame2):
    """Detect quality drops between frames"""
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        sharp1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
        sharp2 = cv2.Laplacian(gray2, cv2.CV_64F).var()
        
        # Normalize
        sharp1 = max(sharp1, 1.0)
        sharp2 = max(sharp2, 1.0)
        
        # Calculate relative change
        quality_change = abs(sharp1 - sharp2) / max(sharp1, sharp2)
        
        return quality_change
        
    except Exception as e:
        return 0.0


def check_color_shift(frame1, frame2):
    """Detect sudden color/lighting shifts"""
    try:
        # Convert to LAB color space
        lab1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2LAB)
        
        # Calculate mean LAB values
        mean1 = np.mean(lab1, axis=(0, 1))
        mean2 = np.mean(lab2, axis=(0, 1))
        
        # Euclidean distance between color distributions
        color_diff = np.linalg.norm(mean1 - mean2) / 100.0  # Normalize
        
        return float(color_diff)
        
    except Exception as e:
        return 0.0


def check_structural_change(frame1, frame2):
    """Check structural similarity between frames"""
    try:
        # Resize for faster processing
        small1 = cv2.resize(frame1, (128, 128))
        small2 = cv2.resize(frame2, (128, 128))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold and calculate percentage of changed pixels
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        change_ratio = np.sum(thresh > 0) / thresh.size
        
        return float(change_ratio)
        
    except Exception as e:
        return 0.0


def get_boundary_weighted_scores(frame_scores, boundary_indices, weight_multiplier=2.0):
    """
    Weight boundary frame scores higher
    
    Args:
        frame_scores: List of scores for all frames
        boundary_indices: Indices of boundary frames
        weight_multiplier: How much to weight boundary frames (default 2x)
    
    Returns:
        float: Weighted average score
    """
    if not frame_scores:
        return 0.5
    
    weighted_scores = []
    weights = []
    
    boundary_set = set(boundary_indices) if boundary_indices else set()
    
    for idx, score in enumerate(frame_scores):
        if idx in boundary_set:
            # Boundary frames get higher weight
            weights.append(weight_multiplier)
        else:
            weights.append(1.0)
        
        weighted_scores.append(score)
    
    # Calculate weighted average
    total_weight = sum(weights)
    if total_weight == 0:
        return np.mean(frame_scores)
    
    weighted_avg = sum(s * w for s, w in zip(weighted_scores, weights)) / total_weight
    
    return float(weighted_avg)
