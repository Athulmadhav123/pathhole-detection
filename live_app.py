import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load your trained model
model = tf.keras.models.load_model("pothole_model.h5")

# Prediction function for each frame
def predict_frame(frame):
    try:
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = img.reshape(1, 224, 224, 3)

        pred = model.predict(img)[0][0]

        if pred > 0.5:
            label = f"POTHOLE ({pred:.2f})"
            color = (0, 0, 255)
        else:
            label = f"NORMAL ({1-pred:.2f})"
            color = (0, 255, 0)

        # Put text on frame
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        return frame

    except Exception as e:
        return frame

# Gradio Interface (LIVE STREAM)
app = gr.Interface(
    fn=predict_frame,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=gr.Image(),
    title="🚀 Live Pothole Detection",
    description="Real-time pothole detection using webcam"
)

app.launch(share=True)