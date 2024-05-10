import streamlit as st
from PIL import Image
import torch
import os
import torchvision.transforms as T
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Set up Detectron2 config
cfg = get_cfg()
cfg.OUTPUT_DIR = './output'
cfg.MODEL.DEVICE = 'cpu'  # Use CPU for inference
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for segmentations
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

# Load Detectron2 model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Streamlit app title
st.title("Object Segmentation using Detectron2")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Perform segmentation and display result
if uploaded_image is not None:
    # Load image
    image = Image.open(uploaded_image)
    
    # Perform segmentation
    outputs = predictor(image)
    
    # Visualize segmentation
    v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    segmented_image = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Display original and segmented images
    st.image([image, segmented_image.get_image()], caption=["Original Image", "Segmented Image"], width=300)
