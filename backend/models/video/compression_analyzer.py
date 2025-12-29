"""
Per-Region Compression Analysis - LAYER 3B
Detects face-swap artifacts by comparing compression levels between regions
"""
import cv2
import numpy as np
from scipy import fftpack


def analyze_region_compression(frame_paths):
    """
    Compare compression artifacts between face and background regions
    
    Args:
        frame_paths: List of frame paths
    
    Returns:
        dict: {
            'score': float (0-1, higher = more suspicious),
            'compression_mismatches': int,
            'avg_face_compression': float,
            'avg_background_compression': float,
            'suspicious_frames': list
        }
    """
    try:
        results = {
            'score': 0.0,
            'compression_mismatches': 0,
            'avg_face_compression': 0.0,
            'avg_background_compression': 0.0,
            'compression_differences': [],
            'suspicious_frames': []
        }
        
        face_compressions = []
        bg_compressions = []
        
        # Analyze subset of frames (every 5th frame for performance)
        sample_indices = range(0, len(frame_paths), max(1, len(frame_paths) // 20))
        
        for idx in sample_indices:
            if idx >= len(frame_paths):
                break
            
            frame_path = frame_paths[idx]
            
            # Detect face and background regions
            face_region, bg_region = extract_face_and_background(frame_path)
            
            if face_region is None or bg_region is None:
                continue
            
            # Analyze compression artifacts in each region
            face_comp = analyze_compression_artifacts(face_region)
            bg_comp = analyze_compression_artifacts(bg_region)
            
            face_compressions.append(face_comp)
            bg_compressions.append(bg_comp)
            
            # Calculate difference
            comp_diff = abs(face_comp - bg_comp)
            results['compression_differences'].append(comp_diff)
            
            # Flag if significant mismatch
            if comp_diff > 0.25:  # Threshold for mismatch
                results['compression_mismatches'] += 1
                results['suspicious_frames'].append({
                    'frame_index': idx,
                    'face_compression': float(face_comp),
                    'bg_compression': float(bg_comp),
                    'difference': float(comp_diff)
                })
        
        # Calculate averages
        if face_compressions:
            results['avg_face_compression'] = float(np.mean(face_compressions))
            results['avg_background_compression'] = float(np.mean(bg_compressions))
            
            # Overall compression mismatch score
            avg_diff = np.mean(results['compression_differences'])
            mismatch_rate = results['compression_mismatches'] / len(face_compressions)
            
            # Combine metrics
            results['score'] = min((avg_diff * 2.0) + (mismatch_rate * 0.5), 1.0)
        
        return results
        
    except Exception as e:
        print(f"Region compression analysis error: {e}")
        return {
            'score': 0.0,
            'error': str(e)
        }


def extract_face_and_background(frame_path):
    """Extract face region and background region from frame"""
    try:
        image = cv2.imread(frame_path)
        if image is None:
            return None, None
        
        h, w = image.shape[:2]
        
        # Use OpenCV directly - more reliable and faster
        return extract_face_opencv(image)
        
    except Exception as e:
        print(f"Face/background extraction error: {e}")
        # Fallback: simple center vs edges
        try:
            image = cv2.imread(frame_path)
            h, w = image.shape[:2]
            face_region = image[h//4:3*h//4, w//4:3*w//4]
            bg_region = np.vstack([image[:h//4, :], image[3*h//4:, :]])
            return face_region, bg_region
        except:
            return None, None


def extract_face_opencv(image):
    """OpenCV fallback for face extraction"""
    try:
        h, w = image.shape[:2]
        
        # Use Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        if len(faces) == 0:
            # No face detected, use center region
            face_region = image[h//4:3*h//4, w//4:3*w//4]
            bg_region = np.vstack([image[:h//4, :], image[3*h//4:, :]])
            return face_region, bg_region
        
        # Use largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        
        # Ensure within bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(w, x + fw)
        y2 = min(h, y + fh)
        
        # Extract face region
        face_region = image[y:y2, x:x2]
        
        # Extract background regions
        top_bg = image[:y, :] if y > 0 else None
        bottom_bg = image[y2:, :] if y2 < h else None
        left_bg = image[y:y2, :x] if x > 0 else None
        right_bg = image[y:y2, x2:] if x2 < w else None
        
        bg_parts = [bg for bg in [top_bg, bottom_bg, left_bg, right_bg] if bg is not None and bg.size > 0]
        
        if not bg_parts:
            return face_region, None
        
        if len(bg_parts) >= 2:
            bg_region = np.vstack(bg_parts[:2])
        else:
            bg_region = bg_parts[0]
        
        return face_region, bg_region
        
    except Exception as e:
        # Final fallback
        h, w = image.shape[:2]
        face_region = image[h//4:3*h//4, w//4:3*w//4]
        bg_region = np.vstack([image[:h//4, :], image[3*h//4:, :]])
        return face_region, bg_region


def analyze_compression_artifacts(region):
    """
    Analyze compression artifacts in a region using DCT
    
    Returns:
        float: Compression artifact score (0-1)
    """
    try:
        if region is None or region.size == 0:
            return 0.5
        
        # Resize to standard size for comparison
        region = cv2.resize(region, (128, 128))
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Apply DCT
        dct = fftpack.dct(fftpack.dct(gray.T, norm='ortho').T, norm='ortho')
        
        # Analyze high-frequency components (compression artifacts)
        h, w = dct.shape
        
        # Divide into blocks (8x8 like JPEG)
        block_size = 8
        block_variances = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = dct[i:i+block_size, j:j+block_size]
                
                # High-frequency components (bottom-right of DCT block)
                high_freq = block[block_size//2:, block_size//2:]
                
                # Calculate energy in high frequencies
                hf_energy = np.sum(np.abs(high_freq))
                block_variances.append(hf_energy)
        
        if not block_variances:
            return 0.5
        
        # Calculate statistics
        mean_energy = np.mean(block_variances)
        std_energy = np.std(block_variances)
        
        # Compression artifact score
        # High mean + low std = uniform compression (typical)
        # Low mean = heavy compression
        # High std = non-uniform compression (suspicious for face-swaps)
        
        # Normalize
        compression_score = (std_energy / (mean_energy + 1.0)) / 10.0
        compression_score = min(compression_score, 1.0)
        
        return float(compression_score)
        
    except Exception as e:
        print(f"Compression artifact analysis error: {e}")
        return 0.5


def detect_blocking_artifacts(region):
    """
    Detect JPEG blocking artifacts
    Face-swaps often have different blocking patterns than background
    """
    try:
        if region is None or region.size == 0:
            return 0.5
        
        # Resize
        region = cv2.resize(region, (128, 128))
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Detect edges (blocking artifacts show up as grid patterns)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal and vertical lines (blocking)
        horizontal_kernel = np.ones((1, 8), np.uint8)
        vertical_kernel = np.ones((8, 1), np.uint8)
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count blocking artifacts
        h_count = np.sum(horizontal_lines > 0)
        v_count = np.sum(vertical_lines > 0)
        
        blocking_score = (h_count + v_count) / edges.size
        
        return float(blocking_score)
        
    except Exception as e:
        return 0.5
