"""DEPRECATED: Use ensemble_detector.py instead.

This file contains an untrained ResNet18 model and should not be used.
The ensemble_detector.py provides much better accuracy with pre-trained models.
"""

import warnings

def predict_image(image):
    """Legacy function - redirects to ensemble detector"""
    warnings.warn(
        "predict_image from image_detector.py is deprecated. "
        "Use predict_ensemble from ensemble_detector.py instead.",
        DeprecationWarning
    )
    from models.ensemble_detector import predict_ensemble
    result = predict_ensemble(image)
    return result.get('score', 0.5)
