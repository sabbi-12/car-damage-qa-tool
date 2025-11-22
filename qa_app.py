import streamlit as st
import cv2
from PIL import Image
import pandas as pd
import glob
import os
import json

# -------------------------
# CONFIG
# -------------------------
IMAGE_FOLDER = "C:\\Users\\Syed Sabbi\\New Project 1\\car dataset\\train"  # path to your train images
ANNOTATION_FILE = os.path.join(IMAGE_FOLDER, "COCO_train_annos.json")
OUTPUT_CSV = "labels.csv"

# -------------------------
# LOAD ANNOTATIONS
# -------------------------
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

# Map image_id to file name
image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

# Map image_id to bounding boxes
image_id_to_boxes = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    bbox = ann["bbox"]  # [x, y, width, height]
    if img_id not in image_id_to_boxes:
        image_id_to_boxes[img_id] = []
    image_id_to_boxes[img_id].append(bbox)

# Get sorted list of image ids
image_ids = sorted(image_id_to_file.keys())

# -------------------------
# Load existing CSV if exists
# -------------------------
if os.path.exists(OUTPUT_CSV):
    df_labels = pd.read_csv(OUTPUT_CSV)
else:
    df_labels = pd.DataFrame(columns=["image", "label"])

# -------------------------
# APP STATE
# -------------------------
if "index" not in st.session_state:
    st.session_state.index = 0

# -------------------------
# FUNCTIONS
# -------------------------
def show_image_with_boxes(img_path, boxes):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Draw boxes
    for box in boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return Image.fromarray(img)

def save_label(image_name, label):
    global df_labels
    if image_name in df_labels["image"].values:
        df_labels.loc[df_labels["image"] == image_name, "label"] = label
    else:
        df_labels.loc[len(df_labels)] = [image_name, label]
    df_labels.to_csv(OUTPUT_CSV, index=False)

# -------------------------
# MAIN APP
# -------------------------
st.title("Car Damage QA Tool")

current_index = st.session_state.index
current_image_id = image_ids[current_index]
current_image_file = image_id_to_file[current_image_id]
current_boxes = image_id_to_boxes.get(current_image_id, [])

img_path = os.path.join(IMAGE_FOLDER, current_image_file)
st.image(show_image_with_boxes(img_path, current_boxes), caption=current_image_file, use_column_width=True)

# Radio button to mark as correct/wrong
label = st.radio("Mark prediction:", ["Correct", "Wrong"])

# Save button
if st.button("Save Label"):
    save_label(current_image_file, label)
    st.success(f"Saved label for {current_image_file}")

# Navigation buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Previous") and st.session_state.index > 0:
        st.session_state.index -= 1
        st.experimental_rerun()
with col2:
    st.write(f"Image {current_index + 1} of {len(image_ids)}")
with col3:
    if st.button("Next") and st.session_state.index < len(image_ids) - 1:
        st.session_state.index += 1
        st.experimental_rerun()
 