"""
Temporal Consistency Analyzer - LAYER 2A
Analyzes temporal consistency across frames:
- Facial landmark stability
- Identity persistence
- Motion smoothness
"""
import cv2
import numpy as np
from PIL import Image
import torch


def analyze_temporal_consistency(frame_paths, timestamps):
    """
    Analyze temporal consistency across frames
    
    Returns:
        dict: {
            'score': float (0-1, higher = more suspicious),
            'landmark_jitter': float,
            'identity_shifts': int,
            'motion_smoothness': float,
            'optical_flow_anomalies': int
        }
    """
    try:
        results = {
            'score': 0.0,
            'landmark_jitter': 0.0,
            'identity_shifts': 0,
            'motion_smoothness': 1.0,
            'optical_flow_anomalies': 0,
            'inconsistencies': []
        }
        
        if len(frame_paths) < 2:
            return results
        
        # 1. Facial landmark tracking
        landmark_stability = analyze_landmark_stability(frame_paths)
        results['landmark_jitter'] = landmark_stability['jitter_score']
        
        if landmark_stability['jitter_score'] > 0.6:
            results['inconsistencies'].append('High facial landmark jitter detected')
        
        # 2. Identity persistence check
        identity_check = check_identity_persistence(frame_paths)
        results['identity_shifts'] = identity_check['num_shifts']
        
        if identity_check['num_shifts'] > 0:
            results['inconsistencies'].append(f'{identity_check["num_shifts"]} identity shifts detected')
        
        # 3. Optical flow analysis
        flow_analysis = analyze_optical_flow(frame_paths)
        results['motion_smoothness'] = flow_analysis['smoothness']
        results['optical_flow_anomalies'] = flow_analysis['anomalies']
        
        if flow_analysis['anomalies'] > len(frame_paths) * 0.2:
            results['inconsistencies'].append('Irregular motion patterns detected')
        
        # Calculate overall score
        score = 0.0
        score += landmark_stability['jitter_score'] * 0.35
        score += min(identity_check['num_shifts'] * 0.3, 1.0) * 0.35
        score += (1.0 - flow_analysis['smoothness']) * 0.30
        
        results['score'] = min(score, 1.0)
        
        return results
        
    except Exception as e:
        print(f"Temporal consistency analysis error: {e}")
        return {
            'score': 0.5,
            'error': str(e)
        }


def analyze_landmark_stability(frame_paths):
    """Analyze facial landmark stability across frames"""
    try:
        # Try MediaPipe Tasks API (new way)
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import mediapipe as mp
            import os
            
            # Path to face landmarker model
            model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models_cache')
            face_landmarker_model = os.path.join(model_dir, 'face_landmarker.task')
            
            if not os.path.exists(face_landmarker_model):
                raise Exception("Face landmarker model not found")
            
            # Proper initialization matching face_analyzer.py
            base_options = python.BaseOptions(
                model_asset_path=face_landmarker_model,
                delegate=python.BaseOptions.Delegate.CPU
            )
            
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            landmarker = vision.FaceLandmarker.create_from_options(options)
            
            landmarks_sequence = []
            
            for frame_path in frame_paths:
                image = cv2.imread(frame_path)
                if image is None:
                    continue
                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                try:
                    detection_result = landmarker.detect(mp_image)
                except Exception:
                    continue
                
                if detection_result.face_landmarks:
                    face_landmarks = detection_result.face_landmarks[0]
                    
                    # Key landmark indices
                    key_indices = [33, 133, 362, 263, 1, 61, 291, 199]  # Eyes, nose, mouth corners
                    
                    key_landmarks = []
                    for idx in key_indices:
                        if idx < len(face_landmarks):
                            lm = face_landmarks[idx]
                            key_landmarks.append([lm.x, lm.y, lm.z])
                    
                    if key_landmarks:
                        landmarks_sequence.append(np.array(key_landmarks))
            
            landmarker.close()
            
            if len(landmarks_sequence) < 2:
                return {'jitter_score': 0.5, 'has_faces': False}
            
            # Calculate jitter (movement between consecutive frames)
            jitters = []
            for i in range(len(landmarks_sequence) - 1):
                diff = np.linalg.norm(landmarks_sequence[i] - landmarks_sequence[i+1])
                jitters.append(diff)
            
            avg_jitter = np.mean(jitters)
            max_jitter = np.max(jitters)
            jitter_variance = np.var(jitters)
            
            # High jitter or variance = suspicious
            jitter_score = min((avg_jitter * 50) + (jitter_variance * 100) + (max_jitter * 20), 1.0)
            
            return {
                'jitter_score': float(jitter_score),
                'avg_jitter': float(avg_jitter),
                'max_jitter': float(max_jitter),
                'has_faces': True
            }
            
        except Exception as mp_error:
            # MediaPipe failed, use OpenCV fallback
            return analyze_landmark_stability_opencv(frame_paths)
        
    except Exception as e:
        print(f"Landmark stability error: {e}")
        return {'jitter_score': 0.5, 'error': str(e)}


def analyze_landmark_stability_opencv(frame_paths):
    """OpenCV fallback for landmark stability"""
    try:
        # Use optical flow as proxy for facial movement
        movements = []
        prev_frame = None
        
        for frame_path in frame_paths:
            image = cv2.imread(frame_path)
            if image is None:
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # Focus on center region (where face likely is)
                h, w = flow.shape[:2]
                center_flow = flow[h//4:3*h//4, w//4:3*w//4]
                
                magnitude = np.sqrt(center_flow[..., 0]**2 + center_flow[..., 1]**2)
                movements.append(np.mean(magnitude))
            
            prev_frame = gray
        
        if len(movements) < 2:
            return {'jitter_score': 0.5, 'has_faces': False}
        
        # Calculate jitter from movement variance
        jitter_score = min(np.var(movements) / 10.0, 1.0)
        
        return {
            'jitter_score': float(jitter_score),
            'has_faces': False
        }
        
    except Exception as e:
        return {'jitter_score': 0.5, 'error': str(e)}


def check_identity_persistence(frame_paths):
    """Check if face identity remains consistent"""
    try:
        # Use FaceNet for face embeddings
        from facenet_pytorch import InceptionResnetV1, MTCNN
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        mtcnn = MTCNN(keep_all=False, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        embeddings = []
        
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            
            # Detect face
            face = mtcnn(img)
            
            if face is not None:
                face = face.unsqueeze(0).to(device)
                
                # Get embedding
                with torch.no_grad():
                    embedding = resnet(face)
                
                embeddings.append(embedding.cpu().numpy().flatten())
        
        if len(embeddings) < 2:
            return {'num_shifts': 0, 'has_faces': False}
        
        # Calculate cosine similarity between consecutive embeddings
        identity_shifts = 0
        similarities = []
        
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i+1]
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(similarity)
            
            # Threshold: similarity < 0.85 indicates identity shift
            if similarity < 0.85:
                identity_shifts += 1
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        
        return {
            'num_shifts': identity_shifts,
            'avg_similarity': float(avg_similarity),
            'has_faces': True
        }
        
    except Exception as e:
        print(f"Identity persistence check error: {e}")
        # Fallback: simple visual similarity
        return check_identity_persistence_fallback(frame_paths)


def check_identity_persistence_fallback(frame_paths):
    """Fallback identity check using simple visual similarity"""
    try:
        identity_shifts = 0
        
        prev_hist = None
        
        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            if img is None:
                continue
            
            # Simple histogram comparison
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Compare histograms
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                if correlation < 0.7:
                    identity_shifts += 1
            
            prev_hist = hist
        
        return {
            'num_shifts': identity_shifts,
            'has_faces': False
        }
        
    except Exception as e:
        print(f"Fallback identity check error: {e}")
        return {'num_shifts': 0, 'has_faces': False}


def analyze_optical_flow(frame_paths):
    """Analyze optical flow for motion smoothness"""
    try:
        anomalies = 0
        flow_magnitudes = []
        
        prev_frame = None
        
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # Calculate magnitude
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                avg_magnitude = np.mean(magnitude)
                flow_magnitudes.append(avg_magnitude)
                
                # Check for anomalies (sudden large changes)
                if len(flow_magnitudes) > 1:
                    if abs(avg_magnitude - flow_magnitudes[-2]) > 5.0:
                        anomalies += 1
            
            prev_frame = gray
        
        if len(flow_magnitudes) < 2:
            return {'smoothness': 1.0, 'anomalies': 0}
        
        # Calculate smoothness
        flow_variance = np.var(flow_magnitudes)
        smoothness = max(0.0, 1.0 - (flow_variance / 100.0))
        
        return {
            'smoothness': float(smoothness),
            'anomalies': anomalies,
            'avg_flow_magnitude': float(np.mean(flow_magnitudes))
        }
        
    except Exception as e:
        print(f"Optical flow analysis error: {e}")
        return {'smoothness': 1.0, 'anomalies': 0, 'error': str(e)}
