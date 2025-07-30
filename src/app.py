import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils.utils import *
from config.constants import N
from sklearn.decomposition import PCA


st.title("Reverse Image Search Engine")

project_root = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.join(project_root, "..", "features_data")
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

features = np.load(os.path.join(features_dir, "features.npy"))
filenames = np.load(os.path.join(features_dir, "filenames.npy"), allow_pickle=True)
class_ids = np.load(os.path.join(features_dir, "class_ids.npy"))


distance_metric = st.radio("Choose a distance metric", options=["euclidean", "cosine"])
use_pca = st.radio("Reducing feature to 100 dims. Apply PCA?", options=["No", "Yes"])

if use_pca == "Yes":
    num_feature_dimensions = 100
    pca = PCA(n_components=num_feature_dimensions)
    pca.fit(features)
    features_to_use = pca.transform(features)
else:
    features_to_use = features


_, feature_extractor = define_model()
knn_model, _, _ = apply_knn(features_to_use, n=N, metric=distance_metric)


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #st.image(uploaded_file, caption="Query Image", use_container_width=True, width=150)

    query_img_path = os.path.join(project_root, "temp_query.png")
    with open(query_img_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Extracting features and finding similar images..."):
        custom_feature = extract_features(query_img_path, feature_extractor).reshape(1, -1)

        if use_pca == "Yes":
            custom_feature = pca.transform(custom_feature)

        distances, indices = knn_model.kneighbors(custom_feature)
        similar_image_paths = [query_img_path] + [filenames[i] for i in indices[0][1:4]]

    fig = plot_images_for_streamlit(similar_image_paths, distances[0])
    st.success(f"Showing similar images using {distance_metric} distance and PCA: {use_pca}")
    st.pyplot(fig)