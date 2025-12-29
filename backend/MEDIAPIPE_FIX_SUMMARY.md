# MediaPipe Fix Summary

## Issue Fixed
The video deepfake detection code in `physiological_analyzer.py` had MediaPipe initialization issues that were causing failures.

## Root Cause
The code was using **incorrect MediaPipe initialization**:

### ❌ **Problem 1: Invalid FaceDetector initialization**
```python
# WRONG - model_asset_buffer=None causes failure
base_options = python.BaseOptions(model_asset_buffer=None)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)
```

### ❌ **Problem 2: Missing FaceLandmarker options**
```python
# INCOMPLETE - Missing critical options
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5
)
```

## Solution Applied
Fixed both issues by referencing the **working code from `face_analyzer.py`**:

### ✅ **Fix 1: Removed broken FaceDetector**
- Replaced MediaPipe FaceDetector with **OpenCV fallback** (more reliable for video)
- OpenCV Haar Cascade is faster and more stable for batch video frame processing

### ✅ **Fix 2: Corrected FaceLandmarker initialization**
```python
# CORRECT - Proper initialization matching face_analyzer.py
base_options = python.BaseOptions(
    model_asset_path=face_landmarker_model,  # Uses actual model file
    delegate=python.BaseOptions.Delegate.CPU
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_face_blendshapes=False,           # Added
    output_facial_transformation_matrixes=False,  # Added
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,        # Added
    min_tracking_confidence=0.5              # Added
)
```

## Changes Made

### File Modified: `models/video/physiological_analyzer.py`

#### Change 1: `extract_face_regions()` function
- **Before**: Attempted to use MediaPipe FaceDetector with `model_asset_buffer=None`
- **After**: Directly uses OpenCV fallback (more reliable for video frames)

#### Change 2: `analyze_blink_pattern()` function
- **Before**: Incomplete FaceLandmarker options
- **After**: Complete initialization matching the working `face_analyzer.py`
- Added missing parameters: `output_face_blendshapes`, `output_facial_transformation_matrixes`, `min_face_presence_confidence`, `min_tracking_confidence`
- Added better error handling with try-catch around individual frame detection

## Key Lessons from `face_analyzer.py`

The working image detector code showed us:

1. **Never use `model_asset_buffer=None`** - Always provide `model_asset_path` with actual file
2. **Include all optional parameters** - MediaPipe requires explicit False values for unused features
3. **Use OpenCV as reliable fallback** - More stable for bulk processing
4. **Proper error handling** - Wrap each detection call in try-catch

## Testing
The fix ensures:
- ✅ No MediaPipe initialization errors
- ✅ Graceful fallback to OpenCV when MediaPipe unavailable
- ✅ Proper error messages for debugging
- ✅ Same API interface (no breaking changes)

## Reference
- Working code: `models/face_analyzer.py` (lines 84-126)
- Fixed code: `models/video/physiological_analyzer.py`
