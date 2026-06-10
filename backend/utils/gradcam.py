# backend/utils/gradcam.py
import cv2
import numpy as np
import tensorflow as tf

# No inner model needed anymore — Integrated Gradients works
# directly on the outer model without sub-model extraction.

def load_inner_model(inner_model_path: str = None):
    """No-op kept for API compatibility with app.py startup."""
    print('  Score-CAM (Integrated Gradients): ready — no inner model needed.')


def generate_scorecam(model, img_array: np.ndarray,
                      class_idx: int, n_steps: int = 50) -> np.ndarray:
    """
    Integrated Gradients attribution map.

    Matches the notebook implementation exactly (Cell 6).
    Fully compatible with MobileNetV3 + Keras 3 — no sub-model access needed.

    Parameters
    ----------
    model      : loaded Keras MobileNetV3 model
    img_array  : (1, 260, 260, 3) float32, values 0-255
    class_idx  : DR class index 0-4
    n_steps    : interpolation steps (more = more accurate, slower)

    Returns
    -------
    heatmap : (H, W) float32 array, values 0.0–1.0
    """
    img      = tf.cast(img_array, tf.float32)        # (1, 260, 260, 3)
    baseline = tf.zeros_like(img)                    # black baseline

    # Interpolated images: baseline → actual image
    alphas     = tf.linspace(0.0, 1.0, n_steps)
    interp     = [baseline + a * (img - baseline) for a in alphas]
    interp_arr = tf.concat(interp, axis=0)           # (n_steps, 260, 260, 3)

    # Gradients for all steps in one pass
    with tf.GradientTape() as tape:
        tape.watch(interp_arr)
        preds       = model(interp_arr, training=False)
        class_preds = preds[:, class_idx]

    grads = tape.gradient(class_preds, interp_arr)   # (n_steps, 260, 260, 3)

    # Trapezoidal average
    avg_grads = tf.reduce_mean(grads, axis=0)        # (260, 260, 3)

    # Attribution = avg_grads × (image − baseline)
    attribution = avg_grads * (img[0] - baseline[0]) # (260, 260, 3)

    # Collapse channels → saliency map
    heatmap = tf.reduce_sum(
        tf.abs(attribution), axis=-1).numpy()        # (260, 260)

    # Normalise to 0–1
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)

    # Smooth
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

    # Re-normalise after blur
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap


def overlay_heatmap(original_rgb: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Unchanged — keeps the same JET colormap overlay logic."""
    if heatmap is None:
        return original_rgb.copy()

    H, W = original_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = (
        alpha * heatmap_colored.astype(np.float32) +
        (1 - alpha) * original_rgb.astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)