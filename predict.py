import os
from datetime import datetime
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
import google.generativeai as genai
from fpdf import FPDF
from werkzeug.utils import secure_filename

# ─── Configuration ──────────────────────────────────────────────────────────────

# Flask setup
app = Flask(__name__)
CORS(app)  # enable CORS for all routes
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini API
genai.configure(api_key="AIzaSyCc7orrlpozTChPcGL98_WX_3og6Q3XooE")
gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# ECG Keras model
ECG_MODEL_PATH = "ecg_lstm_model.h5"
ecg_model = tf.keras.models.load_model(ECG_MODEL_PATH, compile=False)
ecg_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ResNet50 feature extractor
IMAGE_SIZE = (224, 224)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
base_model.trainable = False

# Class labels
CLASS_NAMES = ["arrhythmia", "hmi", "mi", "normal"]

class CustomPDF(FPDF):
    """
    Subclass of FPDF that automatically adds a watermark to every page,
    an icon on the first page, and a footer disclaimer.
    It also sets generous margins to prevent content from colliding with header/footer.
    """
    def __init__(self, watermark_path: str, icon_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.watermark_path = watermark_path
        self.icon_path = icon_path
        # Set custom margins: left, top, right
        self.set_left_margin(15)
        self.set_right_margin(15)
        self.set_top_margin(20)
        # Reserve 25mm at bottom for footer
        self.set_auto_page_break(True, margin=25)

    def header(self):
        # Watermark (centered) on each page
        if os.path.exists(self.watermark_path):
            w = 240  # Set a manual width in mm
            x = (self.w - w) / 2
            y = (self.h - w) / 2
            try:
                self.image(self.watermark_path, x=x, y=y, w=w)
            except RuntimeError:
                pass  # ignore if image fails
        # Icon only on the first page
        if self.page_no() == 1 and os.path.exists(self.icon_path):
            icon_w = 12
            x = self.w - self.r_margin - icon_w
            y = self.t_margin
            try:
                self.image(self.icon_path, x=x, y=y, w=icon_w)
            except RuntimeError:
                pass
        # Move cursor below header
        self.set_y(self.t_margin + 10)

    def footer(self):
        # Footer disclaimer
        self.set_y(-20)
        self.set_font("Arial", "I", 9)
        disclaimer = (
            "Note: This ECG report was generated with AI-assisted tools. "
            "Please consult a licensed cardiologist for interpretation and clinical decisions."
        )
        self.multi_cell(0, 5, disclaimer, align='C')
# ─── Helper Functions ────────────────────────────────────────────────────────────

def preprocess_image(path: str) -> np.ndarray:
    """Load image, resize, normalize, extract features via ResNet50."""
    img = load_img(path, target_size=IMAGE_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    feats = base_model.predict(arr, verbose=0)
    return feats.reshape(1, -1, feats.shape[-1])

def llm_chat(prompt: str) -> str:
    """Query Gemini for a text response."""
    return gemini.generate_content(prompt).text

def get_llm_recommendation(label: str,name:str,age: str,gender: str) -> str:
    """Build and send a structured prompt to Gemini."""
    prompt = f"""
    You are a medical assistant generating part of an official ECG diagnostic report for a real patient.

    This report will be rendered in a structured PDF format and reviewed by both patients and physicians. 
    Please ensure the tone is professional and the formatting is clean, readable, and suitable for printing. 
    Avoid using Markdown syntax (such as **bold**). Instead, write section titles in ALL CAPS and they will be rendered in bold font in the PDF. Use plain dashes (-) for bullet points, and ensure each bullet is properly indented under its respective section. Leave a blank line between each section and between bullet points to improve readability.    Leave blank lines between sections and between bullet items.
    Patient Details:
    - Name: {name}
    - Age: {age}
    - Gender: {gender}

    Diagnosis: {label}

    Please provide the following sections using real content only (generic templates with standard spaces like a proffessional report):

    1. **Condition Summary**: A concise, informative definition of the condition. <br> 
    2. **Key Symptoms**: A bullet list of common symptoms typically observed in patients with this condition.
    3. **Causes and Risk Factors**: A bullet list explaining the likely causes and contributing factors.
    4. **Recommended Actions**: Clinical advice for the next steps (tests, referrals, medications).
    5. **Lifestyle Advice**: Preventive or supportive lifestyle tips to manage or reduce the severity of the condition.
    - Tailor this advice to the patient's age. 
    - If the patient is young (under 30), suggest age-appropriate preventive care, habits, or early interventions.
    - If the patient is elderly (over 60), include advice on managing comorbidities, mobility, diet, and regular monitoring.

    Use a clear, professional tone. Do not use placeholders like "[Patient Name]" or "[Date]". This is a real diagnostic report intended for the above patient.
    """


    return llm_chat(prompt)

def generate_pdf(patient: dict, result: dict, img_path: str) -> BytesIO:
    """
    Create a structured ECG report PDF using:
      - patient (name, age, gender)
      - result (label, confidence, recommendation)
      - ECG image
      - watermark on every page via CustomPDF
      - icon on first page
      - footer disclaimer automatically
    """
    watermark_path = "./heart_faint.png"
    icon_path = "./heart.png"

    pdf = CustomPDF(watermark_path=watermark_path, icon_path=icon_path)
    pdf.add_page()

    # Title and Timestamp
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ECG Diagnostic Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, datetime.now().strftime("%d %b %Y, %H:%M:%S"), ln=True, align="C")
    pdf.ln(5)

    # Patient Information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(50, 6, "Name:", border=0)
    pdf.cell(0, 6, sanitize_text(patient.get("name", "")), ln=True)
    pdf.cell(50, 6, "Age / Gender:", border=0)
    pdf.cell(0, 6, f"{sanitize_text(patient.get('age',''))} / {sanitize_text(patient.get('gender',''))}", ln=True)
    pdf.ln(5)

    # Diagnosis
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Diagnosis", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(50, 6, "Condition:", border=0)
    pdf.cell(0, 6, sanitize_text(result.get("label", "")).upper(), ln=True)
    pdf.cell(50, 6, "Confidence:", border=0)
    pdf.cell(0, 6, f"{result.get('confidence', 0):.1f}%", ln=True)
    pdf.ln(5)

    # Clinical Notes & Recommendations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Clinical Notes & Recommendations", ln=True)
    pdf.set_font("Arial", "", 12)
    for line in result.get("recommendation", "").split("\n"):
        if line.strip():
            pdf.multi_cell(0, 6, sanitize_text(line))
            pdf.ln(1)
    pdf.ln(4)

        # ECG Image page (without heart icon / watermark)
    if os.path.exists(img_path):
        # temporarily disable watermark/icon
        # pdf.watermark_path = None
        # pdf.icon_path = None

        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "ECG Image", ln=True, align="C")
        pdf.ln(5)
        usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.image(img_path, x=pdf.l_margin, y=pdf.get_y(), w=usable_w)
        pdf.ln(5)
        # # restore watermark/icon if needed for later pages
        # pdf.watermark_path = watermark_path
        # pdf.icon_path = icon_path

    # Export PDF
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    buffer = BytesIO(pdf_bytes)
    buffer.seek(0)
    return buffer

# ─── Flask Routes ────────────────────────────────────────────────────────────────
def sanitize_text(text: str) -> str:
    """Replace unsupported characters and encode to latin-1."""
    return text.replace('•','-').encode('latin-1', 'replace').decode('latin-1')
@app.route("/", methods=["GET"])
def home():
    return "Backend is working properly...!"

@app.route("/predict", methods=["POST"])
def predict():
    # 1) Validate input
    if "image" not in request.files:
        return jsonify({"error": "Missing `image` field"}), 400

    image = request.files["image"]
    name  = request.form.get("name")
    age   = request.form.get("age")
    gender= request.form.get("gender")

    if not all([name, age, gender]):
        return jsonify({"error": "Missing patient info (name, age, gender)"}), 400

    # 2) Save upload
    filename   = secure_filename(image.filename)
    save_path  = os.path.join(UPLOAD_FOLDER, filename)
    image.save(save_path)

    try:
        # 3) Model inference
        feats = preprocess_image(save_path)
        preds = ecg_model.predict(feats, verbose=0)
        idx   = int(np.argmax(preds))
        label = CLASS_NAMES[idx]
        conf  = float(np.max(preds)) * 100

        # 4) LLM recommendation
        rec  = get_llm_recommendation(label,name,age,gender)

        # 5) Generate PDF with sanitized inputs
        patient = {
            "name": sanitize_text(name),
            "age": sanitize_text(age),
            "gender": sanitize_text(gender)
        }
        result = {
            "label": sanitize_text(label),
            "confidence": conf,
            "recommendation": sanitize_text(rec)
        }
        pdf_io = generate_pdf(patient, result, save_path)

        # 6) Return PDF
        return send_file(
            pdf_io,
            download_name=f"ECG_Report_{name.replace(' ', '_')}.pdf",
            mimetype="application/pdf",
            as_attachment=True
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
