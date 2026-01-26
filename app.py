import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import random
import requests
import zipfile

MODEL_URL = "https://drive.google.com/file/d/1N24Cmzw_IbMg2VpWG2uSkY5oaKH_D0j1"
MODEL_PATH = "snake_classifier_final.pth"

DATASET_URL = "https://drive.google.com/file/d/1XqEY3l0oIBWfT6XNZx7Ydr4MXcSTVhSG"
DATASET_ZIP = "dataset_final.zip"
DATASET_DIR = "dataset_final"
DATASET_TRAIN_DIR = "dataset_final/train"

IMAGE_SIZE = 384
CONFIDENCE_THRESHOLD = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Snake Species Classifier",
    page_icon=":)",
    layout="centered"
)

st.title("Определение вида змеи по фото")
st.markdown(
    "Загрузите фотографию змеи — модель определит **вид** "
    "и покажет **3 примера этого вида** из датасета."
)

def download_model():
    if os.path.exists(MODEL_PATH):
        return

    st.info("Скачивание модели, подождите...")

    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

def download_and_extract_dataset():
    if os.path.exists(DATASET_TRAIN_DIR):
        return

    st.info("Скачивание датасета, подождите (первый запуск может быть долгим)...")

    response = requests.get(DATASET_URL, stream=True)
    with open(DATASET_ZIP, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
        zip_ref.extractall(".")

    os.remove(DATASET_ZIP)

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_to_idx = checkpoint["class_names"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    classes = [idx_to_class[i] for i in range(num_classes)]
    return model, classes

infer_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_species(image):
    x = infer_tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    prob, idx = torch.max(probs, dim=1)
    return classes[idx.item()], float(prob.item())

def get_example_images(species, n=3):
    class_dir = os.path.join(DATASET_TRAIN_DIR, species)

    if not os.path.exists(class_dir):
        return []

    images = [
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    return random.sample(images, min(n, len(images)))

download_model()
download_and_extract_dataset()

model, classes = load_model()

uploaded_file = st.file_uploader(
    "Загрузите изображение змеи",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Загруженное изображение",
        use_column_width=True
    )

    if st.button("Определить вид"):
        with st.spinner("Анализ изображения..."):
            species, confidence = predict_species(image)

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                "Модель не уверена в результате. "
                "Попробуйте другое изображение."
            )
        else:
            st.success(f"**Определённый вид:** `{species}`")
            st.write(f"Уверенность модели: **{confidence:.2%}**")

            st.subheader("Примеры этого вида из датасета")
            example_images = get_example_images(species)

            if example_images:
                cols = st.columns(len(example_images))
                for col, img_path in zip(cols, example_images):
                    col.image(img_path, use_column_width=True)
            else:
                st.info("Нет изображений для этого вида.")

st.markdown("---")
st.markdown(
    "<center>ML-классификация змей</center>",
    unsafe_allow_html=True
)