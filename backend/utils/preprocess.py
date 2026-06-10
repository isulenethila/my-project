# backend/utils/preprocess.py
import cv2
import numpy as np

IMG_SIZE    = (260, 260)
CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

RECOMMENDATIONS = {
    'No_DR':         'No diabetic retinopathy detected. Annual screening recommended.',
    'Mild':          'Mild DR detected. Follow-up in 6-12 months recommended.',
    'Moderate':      'Moderate DR detected. Ophthalmologist consultation recommended within 3 months.',
    'Severe':        'Severe DR detected. Urgent ophthalmologist consultation required.',
    'Proliferate_DR':'Proliferative DR detected. Immediate specialist referral required.',
}


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Applies CLAHE enhancement to retinal image (BGR input, BGR output)."""
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2      = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def preprocess_image(image_bytes: bytes):
    """
    Full preprocessing pipeline for uploaded retinal image.
    FIX: CLAHE is now applied BEFORE resizing — matches notebook's load_and_clahe().

    Returns:
        original_rgb  : original resized image for display (uint8)
        clahe_rgb     : CLAHE enhanced image for display (uint8)
        model_input   : float32 array (1, 260, 260, 3) values 0-255 for model
    """
    nparr   = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError('Could not decode image. Please upload a valid JPG or PNG.')

    # Original for display only (resize, no CLAHE)
    original_resized = cv2.resize(img_bgr, IMG_SIZE)
    original_rgb     = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)

    # CLAHE applied to full-resolution image FIRST, then resize
    clahe_bgr     = apply_clahe(img_bgr)
    clahe_resized = cv2.resize(clahe_bgr, IMG_SIZE)
    clahe_rgb     = cv2.cvtColor(clahe_resized, cv2.COLOR_BGR2RGB)

    # Model input: float32, 0-255 range (model's rescaling layer handles /255 internally)
    model_input = clahe_rgb.astype(np.float32)
    model_input = np.expand_dims(model_input, axis=0)  # (1, 260, 260, 3)

    return original_rgb, clahe_rgb, model_input


def numpy_to_base64(img_array: np.ndarray) -> str:
    """Converts numpy RGB image array to base64 JPEG string for the frontend."""
    import base64
    img_bgr    = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer  = cv2.imencode('.jpg', img_bgr)
    b64_string = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpeg;base64,{b64_string}'