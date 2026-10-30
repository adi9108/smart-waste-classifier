"""Gradio app. Named app.py so Hugging Face Spaces (Gradio SDK) picks it up."""
import gradio as gr

from utils import TIPS, load_model, predict

model, classes = load_model()


def classify(img):
    if img is None:
        return {}, ""
    probs = predict(model, classes, img)
    top = next(iter(probs))
    kind, tip = TIPS.get(top, ("", ""))
    return probs, f"**{top.upper()}** — {kind}\n\n{tip}"


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Waste image"),
    outputs=[gr.Label(num_top_classes=6, label="Prediction"), gr.Markdown(label="Disposal tip")],
    title="♻️ Smart Waste Classification",
    description="Upload a photo of a waste item to classify it and get a disposal tip.",
)

if __name__ == "__main__":
    demo.launch()
