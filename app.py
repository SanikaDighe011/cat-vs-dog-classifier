import numpy as np
import gradio as gr
from tensorflow import keras

model = keras.models.load_model("cat_dog_classifier.keras")

def predict(img):
    img = img.convert("RGB").resize((150, 150))
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    prob_dog = float(model.predict(arr, verbose=0)[0][0])

    if 0.35 < prob_dog < 0.65:
        return {"Not sure (cat and dog look equally likely)": 1.0}
    return {"dog": prob_dog, "cat": 1 - prob_dog}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Cat vs Dog Classifier",
    description="Upload a photo of a single cat or dog.",
)

if __name__ == "__main__":
    demo.launch()