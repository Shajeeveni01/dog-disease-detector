import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

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

# Streamlit UI
st.set_page_config(page_title="Dog Disease Detector 🐾", layout="centered")
st.title("🐶 Dog Disease Detection App")
st.markdown("Upload a dog image and see the predicted skin disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    # st.image(image, caption="Uploaded Dog Image", use_column_width=True)
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

    # Overlay heatmap on image
    img_display = np.array(img_resized)
    superimposed_img = cv2.addWeighted(img_display.astype('uint8'), 0.6, heatmap_color, 0.4, 0)

    st.subheader("📍 Grad-CAM Heatmap")
    # st.image(superimposed_img, caption="Model focus area", use_column_width=True)
    st.image(superimposed_img, caption="Model focus area", use_container_width=True)
