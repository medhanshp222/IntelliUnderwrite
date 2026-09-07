import os
import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image


# ============================================================
# BANK STATEMENT PDF PATH
# ============================================================

PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "salary_slip.pdf"
)


# ============================================================
# 1. CONVERT PDF TO IMAGES
# ============================================================

def pdf_to_images(pdf_path):

    pdf = fitz.open(pdf_path)

    images = []

    for page in pdf:

        pix = page.get_pixmap(
            matrix=fitz.Matrix(3, 3)
        )

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        images.append(img)

    pdf.close()

    return images


# ============================================================
# 2. PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Improve OCR resolution
    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    # Light denoising
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Threshold
    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return threshold


# ============================================================
# 3. OCR EXTRACTION
# ============================================================

def extract_text(pdf_path):

    pages = pdf_to_images(pdf_path)

    all_text = []

    for image in pages:

        processed = preprocess_image(image)

        text = pytesseract.image_to_string(
            processed,
            config="--psm 6"
        )

        all_text.append(text)

    return "\n".join(all_text)


if __name__ == "__main__":

    text = extract_text(PDF_PATH)

    print("\n")
    print("=" * 80)
    print("RAW OCR TEXT")
    print("=" * 80)
    print(text)
    print("=" * 80)