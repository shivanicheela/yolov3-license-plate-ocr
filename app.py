import streamlit as st
from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# Load YOLOv3 model
model = YOLO("best.pt")

# Load PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.title("YOLOv3 + PaddleOCR License Plate Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run YOLOv3
    results = model.predict(source=image_np, conf=0.4)
    result = results[0]

    # Draw boxes
    for box in result.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        plate_crop = image_np[y1:y2, x1:x2]

        # Run OCR on cropped plate
        ocr_result = ocr.ocr(plate_crop, cls=True)
        for line in ocr_result:
            for word in line:
                text = word[1][0]
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.image(image_np, caption="Detected license plate", use_column_width=True)
