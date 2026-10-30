import streamlit as st
from PIL import Image

from utils import TIPS, load_model, predict

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️")
st.title("♻️ Smart Waste Classification")
st.caption("Upload or capture a photo of a waste item and the model tells you what it is and how to dispose of it.")


@st.cache_resource
def get_model():
    return load_model()


try:
    model, classes = get_model()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

tab1, tab2 = st.tabs(["📁 Upload image", "📷 Use camera"])
with tab1:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
with tab2:
    cam = st.camera_input("Take a picture")

src = file or cam
if src:
    img = Image.open(src)
    st.image(img, caption="Your image", use_container_width=True)
    probs = predict(model, classes, img)
    top, conf = next(iter(probs.items()))
    kind, tip = TIPS.get(top, ("", ""))
    st.success(f"Prediction: **{top.upper()}**  ({conf*100:.1f}% confidence)")
    if kind:
        st.info(f"**{kind}** — {tip}")
    st.bar_chart(probs)
