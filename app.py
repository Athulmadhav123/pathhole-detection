import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("pothole_model.h5")

# Prediction function
def predict(image):
    try:
        # Convert image to numpy
        img = np.array(image)

        # Resize
        img = cv2.resize(img, (224, 224))

        # Normalize
        img = img / 255.0

        # Reshape
        img = img.reshape(1, 224, 224, 3)

        # Predict
        pred = model.predict(img)[0][0]

        # Result
        if pred > 0.5:
            result = f"🚨 POTHOLE DETECTED ({pred:.2f})"
        else:
            result = f"✅ NORMAL ROAD ({1 - pred:.2f})"

        return result

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🛣️ Pothole Detection System",
    description="Upload a road image to detect potholes using AI"
)

# Launch app
app.launch(share=True)