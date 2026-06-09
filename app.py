from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import segmentation_models_pytorch as smp

import torch
import cv2
import numpy as np
import base64

import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================================
# CONFIG
# =====================================

IMG_SIZE = 256

# =====================================
# FASTAPI
# =====================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# MODEL
# =====================================

def build_model():

    model = smp.UnetPlusPlus(

        encoder_name="resnet34",

        encoder_weights=None,

        in_channels=3,

        classes=1

    )

    return model

# =====================================
# LOAD MODEL
# =====================================

model = build_model()

model.load_state_dict(
    torch.load(
        "best_deeplabv3_resnet34.pth",
        map_location="cpu"
    )
)

model.eval()

# =====================================
# TRANSFORM
# =====================================

infer_transform = A.Compose([

    A.Normalize(),

    ToTensorV2()

])

# =====================================
# HELPER
# =====================================

def preprocess(image):

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = np.stack(
        [image, image, image],
        axis=-1
    )

    transformed = infer_transform(
        image=image
    )

    image = transformed["image"]

    image = image.unsqueeze(0)

    return image


def image_to_base64(img):

    _, buffer = cv2.imencode(
        ".png",
        img
    )

    return base64.b64encode(
        buffer
    ).decode("utf-8")


def calculate_pollution(mask):

    oil_pixels = np.sum(mask == 1)

    total_pixels = mask.size

    percentage = (
        oil_pixels /
        total_pixels
    ) * 100

    if percentage < 10:

        level = "Rendah"

    elif percentage < 30:

        level = "Sedang"

    else:

        level = "Tinggi"

    return percentage, level

# =====================================
# ROUTE
# =====================================

@app.get("/")
def home():

    return {
        "message": "Oil Spill Detection API Running"
    }

# =====================================
# PREDICT
# =====================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = cv2.imdecode(

        np.frombuffer(
            contents,
            np.uint8
        ),

        cv2.IMREAD_GRAYSCALE

    )

    if image is None:

        return {
            "error": "Gagal membaca gambar"
        }

    original = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    tensor = preprocess(
        image
    )

    with torch.no_grad():

        pred = model(
            tensor
        )

        pred = torch.sigmoid(
            pred
        )

        pred = pred.squeeze()

        pred = pred.cpu().numpy()

    mask = (
        pred > 0.5
    ).astype(np.uint8)

    # =====================================
    # PERSENTASE
    # =====================================

    percentage, level = (
        calculate_pollution(mask)
    )

    # =====================================
    # MASK IMAGE
    # =====================================

    mask_img = (
        mask * 255
    ).astype(np.uint8)

    # =====================================
    # OVERLAY IMAGE
    # =====================================

    overlay = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    overlay[mask == 1] = [
        0,
        0,
        255
    ]

    # =====================================
    # RESPONSE
    # =====================================

    return {

        "mask": image_to_base64(
            mask_img
        ),

        "overlay": image_to_base64(
            overlay
        )

    }