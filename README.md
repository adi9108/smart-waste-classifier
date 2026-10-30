https://huggingface.co/spaces/Adityaaa170904/smart-waste-classifier# ♻️ Smart Waste Classification (ML ISE-2)

MobileNetV2 transfer-learning classifier (6 classes: cardboard, glass, metal, paper, plastic, trash)
with a Streamlit app (`streamlit_app.py`) and a Gradio app (`app.py`).

## A. Run on localhost
1. Install Python 3.10-3.12, open a terminal in this folder.
2. `python -m venv venv` then activate it
   - Windows: `venv\Scripts\activate`    - Mac/Linux: `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Download the TrashNet dataset (Kaggle: search "TrashNet", file `dataset-resized`) and place the 6 class
   folders inside `dataset/` so you get `dataset/plastic/xxx.jpg`, etc.
5. `python train.py`  (about 5-15 min; creates `model/waste_model.pth` + `model/classes.json`)
6. Run an app:
   - Streamlit: `streamlit run streamlit_app.py`  -> http://localhost:8501
   - Gradio:    `python app.py`                   -> http://127.0.0.1:7860

## B. Deploy online (free)
### Hugging Face Spaces (Gradio)
1. huggingface.co -> New Space -> SDK: Gradio, hardware: CPU basic (free), Public.
2. Upload: `app.py`, `utils.py`, `requirements.txt`, and the `model/` folder (waste_model.pth + classes.json).
   (Remove `streamlit` from requirements.txt if you like; not required.)
3. Wait for "Building" -> "Running". Your link: huggingface.co/spaces/<username>/<space-name>

### Streamlit Community Cloud
1. Create a GitHub repo and push: `streamlit_app.py`, `utils.py`, `requirements.txt`, `model/` (not `dataset/`).
2. share.streamlit.io -> New app -> pick repo/branch -> main file `streamlit_app.py` -> Deploy.
