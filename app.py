import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import random
import zipfile
import gdown

# ================== CONFIG ==================

MODEL_ID = "1N24Cmzw_IbMg2VpWG2uSkY5oaKH_D0j1"
MODEL_PATH = "snake_classifier_final.pth"

DATASET_ID = "1XqEY3l0oIBWfT6XNZx7Ydr4MXcSTVhSG"
DATASET_ZIP = "dataset_final.zip"
DATASET_DIR = "dataset_final/train"

IMAGE_SIZE = 384
CONFIDENCE_THRESHOLD = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Snake Species Classifier",
    layout="centered"
)

st.title("Определение вида змеи по фото")
st.write(
    "Загрузите фотографию змеи — модель определит вид "
    "и покажет 3 случайных примера этого вида из датасета."
)

def download_model():
    if os.path.exists(MODEL_PATH):
        return

    st.info("Скачивание модели...")
    gdown.download(
        id=MODEL_ID,
        output=MODEL_PATH,
        quiet=False
    )

def download_and_extract_dataset():
    if os.path.exists(DATASET_DIR):
        return

    st.info("Скачивание датасета (первый запуск может быть долгим)...")

    gdown.download(
        id=DATASET_ID,
        output=DATASET_ZIP,
        quiet=False
    )

    with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
        zip_ref.extractall(".")

    os.remove(DATASET_ZIP)

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    class_to_idx = checkpoint["class_names"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    classes = [idx_to_class[i] for i in range(num_classes)]
    return model, classes

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)

    conf, idx = torch.max(probs, dim=1)
    return classes[idx.item()], float(conf)

def get_examples(species, n=3):
    folder = os.path.join(DATASET_DIR, species)
    if not os.path.exists(folder):
        return []

    images = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    return random.sample(images, min(n, len(images)))

download_model()
download_and_extract_dataset()
model, classes = load_model()

uploaded = st.file_uploader(
    "Загрузите изображение змеи",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Определить вид"):
        species, confidence = predict(image)

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("Модель не уверена в результате")
        else:
            st.success(f"Вид: {species}")
            st.write(f"Уверенность: {confidence:.2%}")

            examples = get_examples(species)
            if examples:
                cols = st.columns(len(examples))
                for c, img in zip(cols, examples):
                    c.image(img, use_column_width=True)
