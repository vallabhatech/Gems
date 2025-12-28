import os
from PIL import Image
import numpy as np
import config
from models.ensemble_detector import predict_ensemble
from models.frequency_analyzer import analyze_frequency_domain
from models.face_analyzer import analyze_face
from models.metadata_analyzer import analyze_metadata


def analyze_image_comprehensive(image_path):
    """
    Comprehensive image analysis using all detection methods.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        dict: Complete analysis results with scores and breakdown
    """
    try:
        # Load image once for reuse
        image = Image.open(image_path).convert('RGB')
        
        results = {
            'neural_network': None,
            'frequency_domain': None,
            'facial_analysis': None,
            'metadata_forensics': None,
            'final_score': 0.0,
            'risk_level': 'Unknown',
            'confidence': 0.0
        }
        
        # 1. Neural Network Ensemble
        if config.NEURAL_ENSEMBLE_ENABLED:
            try:
                neural_result = predict_ensemble(image)
                results['neural_network'] = neural_result
            except Exception as e:
                print(f"Neural network analysis failed: {e}")
                results['neural_network'] = {'score': 0.5, 'error': str(e)}
        
        # 2. Frequency Domain Analysis
        if config.FREQUENCY_ANALYSIS_ENABLED:
            try:
                freq_result = analyze_frequency_domain(image)
                results['frequency_domain'] = freq_result
            except Exception as e:
                print(f"Frequency analysis failed: {e}")
                results['frequency_domain'] = {'score': 0.5, 'error': str(e)}
        
        # 3. Facial Analysis
        if config.FACE_ANALYSIS_ENABLED:
            try:
                face_result = analyze_face(image)
                results['facial_analysis'] = face_result
            except Exception as e:
                print(f"Face analysis failed: {e}")
                results['facial_analysis'] = {'score': 0.5, 'error': str(e)}
        
        # 4. Metadata Forensics
        if config.METADATA_ANALYSIS_ENABLED:
            try:
                metadata_result = analyze_metadata(image_path)
                results['metadata_forensics'] = metadata_result
            except Exception as e:
                print(f"Metadata analysis failed: {e}")
                results['metadata_forensics'] = {'score': 0.5, 'error': str(e)}
        
        # Combine all scores with AGGRESSIVE DYNAMIC WEIGHTING
        final_score, confidence = combine_scores_aggressive(results)
        results['final_score'] = final_score
        results['confidence'] = confidence
        results['risk_level'] = determine_risk_level(final_score)
        
        return results
    
    except Exception as e:
        print(f"Comprehensive analysis error: {e}")
        return {
            'error': str(e),
            'final_score': 0.5,
            'risk_level': 'Unknown'
        }


def combine_scores_aggressive(results):
    """
    AGGRESSIVE dynamic weighting - heavily trusts neural networks when confident.
    
    Returns:
        tuple: (final_score, confidence)
    """
    weights = config.ENSEMBLE_WEIGHTS.copy()
    
    scores = []
    active_weights = []
    confidences = []
    
    # Neural Network - DOMINANT METHOD
    nn_result = results.get('neural_network')
    if nn_result and 'score' in nn_result:
        nn_score = nn_result['score']
        nn_confidence = nn_result.get('confidence', 0.8)
        agreement = nn_result.get('model_agreement', 'unknown')
        
        scores.append(nn_score)
        base_weight = weights['neural']
        
        # AGGRESSIVE BOOST: Very confident + unanimous = 2.5x weight
        if nn_confidence > 0.95 and agreement == 'unanimous':
            base_weight *= 2.5
        # Strong confidence + strong agreement = 2.0x weight
        elif nn_confidence > 0.93 and agreement in ['unanimous', 'strong_agreement']:
            base_weight *= 2.0
        # High confidence = 1.7x weight
        elif nn_confidence > 0.90:
            base_weight *= 1.7
        # Medium-high confidence = 1.4x weight
        elif nn_confidence > 0.85:
            base_weight *= 1.4
        
        active_weights.append(base_weight)
        confidences.append(nn_confidence)
    
    # Frequency Domain
    freq_result = results.get('frequency_domain')
    if freq_result and 'score' in freq_result:
        freq_score = freq_result['score']
        scores.append(freq_score)
        freq_weight = weights['frequency']
        
        # Boost if agrees with neural network
        if nn_result and 'score' in nn_result:
            nn_score = nn_result['score']
            # Both say fake
            if nn_score > 0.7 and freq_score > 0.6:
                freq_weight *= 1.4
            # Both say real
            elif nn_score < 0.3 and freq_score < 0.4:
                freq_weight *= 1.4
        
        active_weights.append(freq_weight)
        confidences.append(0.7)
    
    # Face Analysis - DYNAMIC based on detection
    face_result = results.get('facial_analysis')
    if face_result and 'score' in face_result:
        face_score = face_result['score']
        face_detected = face_result.get('face_detected', False)
        
        if not face_detected:
            # REDISTRIBUTE face weight to neural network
            if config.FACE_NOT_DETECTED_REDISTRIBUTE and len(active_weights) > 0:
                active_weights[0] += weights['face']
        else:
            scores.append(face_score)
            face_weight = weights['face']
            
            # Enhanced weighting based on sub-scores
            eye_score = face_result.get('eye_quality_score', 0.5)
            texture_score = face_result.get('skin_texture_score', 0.5)
            symmetry_score = face_result.get('symmetry_score', 0.5)
            
            # If multiple strong anomalies detected
            high_anomalies = sum([
                eye_score > 0.7,
                texture_score > 0.7,
                symmetry_score > 0.65
            ])
            
            if high_anomalies >= 2:
                # Multiple anomalies = boost face weight
                face_weight *= 1.5
                
                # If neural also says fake, boost even more
                if nn_result and nn_result.get('score', 0) > 0.7:
                    face_weight *= 1.3
            
            active_weights.append(face_weight)
            confidences.append(0.75)
    
    # Metadata
    meta_result = results.get('metadata_forensics')
    if meta_result and 'score' in meta_result:
        meta_score = meta_result['score']
        scores.append(meta_score)
        meta_weight = weights['metadata']
        
        # Boost if clear evidence found
        if meta_result.get('exif_suspicious', False) or meta_result.get('ela_anomalies', False):
            meta_weight *= 1.4
        
        active_weights.append(meta_weight)
        confidences.append(0.65)
    
    if len(scores) == 0:
        return 0.5, 0.0
    
    # CROSS-METHOD AGREEMENT BOOST
    if len(scores) >= 3:
        agreement_count = 0
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                if abs(scores[i] - scores[j]) < 0.15:
                    agreement_count += 1
        
        # If 2+ pairs agree, boost all agreeing methods
        if agreement_count >= 2:
            for idx in range(len(active_weights)):
                active_weights[idx] *= 1.2
    
    # Normalize weights
    total_weight = sum(active_weights)
    normalized_weights = [w / total_weight for w in active_weights]
    
    # Calculate weighted score
    final_score = sum(s * w for s, w in zip(scores, normalized_weights))
    
    # Calculate overall confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
    
    # Boost confidence if low variance (methods agree)
    if len(scores) >= 2:
        score_variance = np.var(scores)
        if score_variance < 0.04:  # Very low variance
            avg_confidence = min(avg_confidence * 1.3, 1.0)
        elif score_variance < 0.08:  # Low variance
            avg_confidence = min(avg_confidence * 1.15, 1.0)
    
    return final_score, avg_confidence


def determine_risk_level(score):
    """Convert score to risk level"""
    thresholds = config.RISK_THRESHOLDS
    
    if score >= thresholds['high']:
        return "High"
    elif score >= thresholds['medium']:
        return "Medium"
    else:
        return "Low"


def generate_detailed_breakdown(results):
    """Generate human-readable breakdown of analysis"""
    breakdown = []
    
    # Neural Network
    if results.get('neural_network'):
        nn = results['neural_network']
        breakdown.append(f"Neural Network Analysis: {nn.get('score', 0.0):.2f}")
        if 'model_agreement' in nn:
            breakdown.append(f"  - Model Agreement: {nn['model_agreement']}")
        if 'num_models' in nn:
            breakdown.append(f"  - Models Used: {nn['num_models']}")
    
    # Frequency Domain
    if results.get('frequency_domain'):
        freq = results['frequency_domain']
        breakdown.append(f"Frequency Analysis: {freq.get('score', 0.0):.2f}")
        if freq.get('fft_anomaly'):
            breakdown.append("  - FFT anomaly detected")
        if freq.get('dct_anomaly'):
            breakdown.append("  - DCT anomaly detected")
    
    # Face Analysis
    if results.get('facial_analysis'):
        face = results['facial_analysis']
        breakdown.append(f"Facial Analysis: {face.get('score', 0.0):.2f}")
        if face.get('face_detected'):
            breakdown.append("  - Face detected")
            if face.get('symmetry_anomaly'):
                breakdown.append("  - Asymmetry detected")
            if face.get('eye_anomaly'):
                breakdown.append("  - Eye quality issues")
            if face.get('texture_anomaly'):
                breakdown.append("  - Unnatural skin texture")
        else:
            breakdown.append("  - No face detected")
    
    # Metadata
    if results.get('metadata_forensics'):
        meta = results['metadata_forensics']
        breakdown.append(f"Metadata Forensics: {meta.get('score', 0.0):.2f}")
        if not meta.get('exif_present'):
            breakdown.append("  - No EXIF data found")
        if meta.get('ela_anomalies'):
            breakdown.append("  - Compression anomalies detected")
        software = meta.get('editing_software_detected', 'Unknown')
        if software != 'Unknown':
            breakdown.append(f"  - Software: {software}")
    
    return "\n".join(breakdown)
