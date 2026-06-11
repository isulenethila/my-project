import os
import io
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import tensorflow as tf
from tensorflow import keras

from utils.preprocess import (
    preprocess_image, numpy_to_base64,
    CLASS_NAMES, RECOMMENDATIONS
)


from utils.gradcam import generate_scorecam, overlay_heatmap, load_inner_model
from utils.report import generate_pdf_report

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title='DR Detection API',
    description='Diabetic Retinopathy Detection using MobileNetV3 + Score-CAM',
    version='1.0.0'
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Load Model Once on Startup ────────────────────────────────────────────────
MODEL_PATH = os.path.join('model', 'best_dr_model.keras')
model = keras.models.load_model(MODEL_PATH, compile=False)

@app.on_event('startup')
async def load_model():
    global model
    print('Loading MobileNetV3 model...')
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print('Model loaded successfully!')
    print(f'  Input : {model.input_shape}')
    print(f'  Output: {model.output_shape}')

    # Load inner model for Score-CAM
    load_inner_model('model/mobilenet_inner.keras')
    print('Score-CAM ready.')

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get('/')
def root():
    return {
        'status': 'running',
        'model':  'MobileNetV3Large',
        'task':   'Diabetic Retinopathy Detection (5 classes)',
    }

@app.get('/health')
def health():
    return {
        'status':       'healthy',
        'model_loaded': model is not None
    }


# ── Main Prediction Endpoint ──────────────────────────────────────────────────
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
        raise HTTPException(
            status_code=400,
            detail='Please upload a JPG or PNG image.'
        )

    # Read and preprocess
    image_bytes = await file.read()
    try:
        original_rgb, clahe_rgb, model_input = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Predict
    probs      = model.predict(model_input, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    pred_name  = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    probabilities = {
        CLASS_NAMES[i]: round(float(probs[i]) * 100, 1)
        for i in range(len(CLASS_NAMES))
    }

    # Generate Score-CAM heatmap (fallback to gradient saliency)
    heatmap     = generate_scorecam(model, model_input, pred_idx)
    overlay     = overlay_heatmap(original_rgb, heatmap)
    # Convert to base64
    original_b64 = numpy_to_base64(original_rgb)
    clahe_b64    = numpy_to_base64(clahe_rgb)
    gradcam_b64  = numpy_to_base64(overlay)

    return {
        'prediction':     pred_name,
        'confidence':     round(confidence, 1),
        'probabilities':  probabilities,
        'recommendation': RECOMMENDATIONS[pred_name],
        'original_image': original_b64,
        'clahe_image':    clahe_b64,
        'gradcam_image':  gradcam_b64,
        'model':          'MobileNetV3Large',
        'preprocessing':  'CLAHE + 260x260',
    }


# ── PDF Report Endpoint ───────────────────────────────────────────────────────
@app.post('/report')
async def generate_report(
    patient_name:  str   = Form(''),
    prediction:    str   = Form(...),
    confidence:    float = Form(...),
    probabilities: str   = Form(...),
    original_b64:  str   = Form(...),
    gradcam_b64:   str   = Form(...),
):
    import json
    probs_dict     = json.loads(probabilities)
    recommendation = RECOMMENDATIONS.get(prediction, '')

    pdf_bytes = generate_pdf_report(
        patient_name   = patient_name,
        prediction     = prediction,
        confidence     = confidence,
        probabilities  = probs_dict,
        original_b64   = original_b64,
        gradcam_b64    = gradcam_b64,
        recommendation = recommendation,
    )

    return Response(
        content    = pdf_bytes,
        media_type = 'application/pdf',
        headers    = {
            'Content-Disposition':
            f'attachment; filename=DR_Report_{prediction}.pdf'
        }
    )


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)