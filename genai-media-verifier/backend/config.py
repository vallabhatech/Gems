# Configuration for Deepfake Detection System
#
# AGGRESSIVE ensemble weights - heavily favors neural networks when confident
ENSEMBLE_WEIGHTS = {
    'neural': 0.50,      # Base weight for neural networks
    'frequency': 0.25,   # Frequency domain analysis
    'face': 0.15,        # Facial analysis (only when face detected)
    'metadata': 0.10     # Metadata forensics
}

# Risk level thresholds
RISK_THRESHOLDS = {
    'high': 0.65,    # More sensitive detection
    'medium': 0.40
}

# Model configuration
MODEL_CONFIG = {
    'huggingface': 'prithivMLmods/Deep-Fake-Detector-Model',
    'device': 'cuda',  # Will auto-fallback to CPU if CUDA unavailable
}

# File upload limits
MAX_FILE_SIZE_MB = 50
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# Analysis settings
FREQUENCY_ANALYSIS_ENABLED = True
FACE_ANALYSIS_ENABLED = True
METADATA_ANALYSIS_ENABLED = True
NEURAL_ENSEMBLE_ENABLED = True

# AGGRESSIVE dynamic weighting settings
ENABLE_DYNAMIC_WEIGHTING = True
NEURAL_CONFIDENCE_BOOST = 2.5      # 2.5x boost when >95% confidence + unanimous
NEURAL_HIGH_CONFIDENCE_BOOST = 2.0  # 2.0x boost when >93% confidence + strong agreement
NEURAL_MEDIUM_BOOST = 1.7          # 1.7x boost when >90% confidence
AGREEMENT_BOOST = 1.4              # 1.4x boost when multiple methods agree
FACE_NOT_DETECTED_REDISTRIBUTE = True  # Redistribute face weight to neural when no face detected

# MediaPipe face detection confidence
FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_SIZE = 50

# Feature toggles
ENABLE_DETAILED_BREAKDOWN = True
ENABLE_CONFIDENCE_SCORES = True
