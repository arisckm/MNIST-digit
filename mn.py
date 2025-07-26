import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import os

# Load model saved in native Keras (.keras) format
model_path = os.path.join(os.path.dirname(__file__), "mnist_model.keras")
model = tf.keras.models.load_model(model_path)

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🧠 MNIST Handwritten Digit Recognizer")

st.sidebar.title("✏️ Input Method")
input_mode = st.sidebar.radio("Choose how to provide a digit:", ("Upload Image", "Draw on Canvas"))

def preprocess_image(pil_image):
    pil_image = pil_image.convert("L")  # Convert to grayscale
    pil_image = ImageOps.invert(pil_image)  # Invert (white digits on black background)

    # Convert to numpy and threshold
    img = np.array(pil_image)
    img = (img > 20).astype(np.uint8) * 255

    # Crop non-zero area
    nonzero = np.argwhere(img)
    if nonzero.size > 0:
        top_left = nonzero.min(axis=0)
        bottom_right = nonzero.max(axis=0)
        img = img[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

    # Resize while maintaining aspect ratio
    h, w = img.shape
    if h > w:
        new_h, new_w = 20, int(w * (20.0 / h))
    else:
        new_w, new_h = 20, int(h * (20.0 / w))
    img = Image.fromarray(img).resize((new_w, new_h), Image.ANTIALIAS)
    img = np.array(img)

    # Pad to 28x28
    padded = np.pad(
        img,
        (((28 - new_h) // 2, (28 - new_h + 1) // 2),
         ((28 - new_w) // 2, (28 - new_w + 1) // 2)),
        mode='constant', constant_values=0
    )

    padded = padded.astype("float32") / 255.0
    st.image(padded, caption="🖼️ Processed Input (28x28)", width=150, clamp=True)
    return padded.reshape(1, 28, 28, 1)

# Upload Image
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image (digit on white background)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", width=150)
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        st.success(f"🎯 Predicted Digit: **{np.argmax(prediction)}**")

# Draw on Canvas
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
