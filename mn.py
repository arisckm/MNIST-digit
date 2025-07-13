import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model("mn.h5")

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🧠 MNIST Handwritten Digit Recognizer")

st.sidebar.title("✏️ Input Method")
input_mode = st.sidebar.radio("Choose how to provide a digit:", ("Upload Image", "Draw on Canvas"))

def preprocess_image(pil_image):
    pil_image = pil_image.convert('L')           # Convert to grayscale
    pil_image = ImageOps.invert(pil_image)        # Invert (white digit on black bg)

    img = np.array(pil_image)

    # Thresholding to clean noise
    _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    # Find digit contour
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        img = img[y:y+h, x:x+w]  # Crop to bounding box

    # Resize maintaining aspect ratio
    h, w = img.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20.0 / h))
    else:
        new_w = 20
        new_h = int(h * (20.0 / w))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    padded = np.pad(img, (((28 - new_h) // 2, (28 - new_h + 1) // 2),
                          ((28 - new_w) // 2, (28 - new_w + 1) // 2)), 'constant', constant_values=0)

    # Normalize
    padded = padded.astype("float32") / 255.0

    st.image(padded, caption="🖼️ Processed Input (28x28)", width=150, clamp=True)

    return padded.reshape(1, 28, 28, 1)

# Upload
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image (digit on white background)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", width=150)
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        st.success(f"🎯 Predicted Digit: **{np.argmax(prediction)}**")

# Draw
else:
    st.write("🎨 Draw a digit (0–9) below:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=12,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
        st.image(img, caption="✏️ Your Drawing", width=150)
        processed = preprocess_image(img)
        prediction = model.predict(processed)
        st.success(f"🎯 Predicted Digit: **{np.argmax(prediction)}**")
