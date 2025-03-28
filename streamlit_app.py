import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load model
model = tf.keras.models.load_model("dog_disease_model.keras")

# Class labels
class_labels = [
    'Bacterial_Infection', 'Conjunctival_Injection_or_Redness', 'Demodicosis', 'Dermatitis',
    'Fungal_Infection', 'Healthy', 'Hypersensitivity', 'Keratosis', 'Malassezia',
    'Nasal_Discharge', 'Ocular_Discharge', 'Pyoderma', 'Skin_Lesions', 'flea_allergy',
    'hotspot', 'mange', 'ringworm'
]

# Grad-CAM utility function
def get_gradcam_heatmap(model, img_array, class_index, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# PDF report generator
def generate_pdf(prediction, confidence, timestamp):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 80, "🐶 Dog Disease Detection Report")

    # Timestamp
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 120, f"Timestamp: {timestamp}")

    # Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 160, f"Prediction: {prediction}")

    # Confidence
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 190, f"Confidence: {confidence:.2f}%")

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 40, "Generated by Dog360 AI Tool")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Streamlit UI
st.set_page_config(page_title="Dog Disease Detector 🐾", layout="centered")
st.title("🐶 Dog Disease Detection App")
st.markdown("Upload a dog image and see the predicted skin disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Dog Image", use_container_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)[0]
    top_index = np.argmax(predictions)
    confidence = predictions[top_index] * 100

    st.subheader("🔍 Prediction")
    st.write(f"**{class_labels[top_index]}** — {confidence:.2f}% confidence")

    if confidence < 50:
        st.warning("⚠️ The model is unsure — confidence is below 50%.")

    # Grad-CAM heatmap
    heatmap = get_gradcam_heatmap(model, img_array, class_index=top_index)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    img_display = np.array(img_resized)
    superimposed_img = cv2.addWeighted(img_display.astype('uint8'), 0.6, heatmap_color, 0.4, 0)

    st.subheader("📍 Grad-CAM Heatmap")
    st.image(superimposed_img, caption="Model focus area", use_container_width=True)

    # 🧾 Report Data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_data = {
        "Prediction": [class_labels[top_index]],
        "Confidence (%)": [f"{confidence:.2f}"],
        "Timestamp": [timestamp]
    }
    report_df = pd.DataFrame(report_data)

    st.subheader("🧾 Report Summary")
    st.dataframe(report_df)

    # CSV download
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Report as CSV",
        data=csv_buffer.getvalue(),
        file_name="dog_disease_report.csv",
        mime="text/csv"
    )

    # PDF download
    pdf_buffer = generate_pdf(
        prediction=class_labels[top_index],
        confidence=confidence,
        timestamp=timestamp
    )
    st.download_button(
        label="📄 Download Report as PDF",
        data=pdf_buffer,
        file_name="dog_disease_report.pdf",
        mime="application/pdf"
    )