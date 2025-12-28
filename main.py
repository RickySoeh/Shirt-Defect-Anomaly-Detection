import streamlit as st
import torch
import timm
import faiss
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
BACKBONE_NAME = "resnet18"
OUT_INDICES = [2] 

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

ASSETS_DIR = "patchcore_assets"
THRESHOLD = float(np.load(f"{ASSETS_DIR}/threshold.npy"))

faiss_index = faiss.read_index(f"{ASSETS_DIR}/faiss_index.bin")
FAISS_DIM = faiss_index.d

@st.cache_resource
def load_model():
    model = timm.create_model(
        BACKBONE_NAME,
        pretrained=True,
        features_only=True,
        out_indices=OUT_INDICES
    )
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def patchcore_predict_pil(image_pil):
    img = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = model(img)[0] 
        B, C, H, W = feats.shape

        feats = feats.permute(0, 2, 3, 1)  
        feats = feats.reshape(-1, C)       
        feats = feats.cpu().numpy().astype(np.float32)

    if feats.shape[1] != FAISS_DIM:
        raise ValueError(
            f"FAISS dim mismatch: got {feats.shape[1]}, expected {FAISS_DIM}"
        )

    D, _ = faiss_index.search(feats, k=1)
    score = float(D.mean())  

    return score

st.title("PatchCore Shirt Defect Detection")

uploaded = st.file_uploader(
    "Upload a shirt image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image")

    with st.spinner("Analyzing image..."):
        score = patchcore_predict_pil(image)

    st.subheader("Result")
    st.write(f"Anomaly Score: **{score:.4f}**")
    st.write(f"Threshold: **{THRESHOLD:.4f}**")

    if score > THRESHOLD:
        st.error("DEFECT")
    else:
        st.success("NORMAL")
