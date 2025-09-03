import numpy as np
import os
import random
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import norm

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from config.constants import *


def define_model():
    base_model = ResNet50(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    input_tensor = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = base_model(input_tensor)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    output_tensor = Dense(NUM_CLASSES, activation='softmax')(x)

    full_model = Model(inputs=input_tensor, outputs=output_tensor)
    return full_model, base_model


def extract_features(img_path, feature_model):
    img = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = feature_model.predict(img_array)
    flattened = features.flatten()
    normalized = flattened / norm(flattened)
    return normalized


def extract_classname_filename(path):
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2] + '/' + parts[-1]
    else:
        return parts[-1]


def plot_images(filenames, distances):
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    plt.figure(figsize=(12, 10))
    columns = 4
    for i, image in enumerate(images):
        ax = plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + extract_classname_filename(filenames[i]))
        else:
            ax.set_title("Similar Image\n" + extract_classname_filename(filenames[i]) + "\nDistance: " + str(float("{0:.2f}".format(distances[i]))))
        plt.axis('off')
        plt.imshow(image)
        plt.savefig("output.png")
    
    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    query_image_name = os.path.basename(filenames[0]).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{query_image_name}_{timestamp}.png")

    plt.savefig(output_path)
    print(f"[INFO] Plot saved to: {output_path}")

    
def apply_knn(features, n=5, metric='cosine'):
  knn = NearestNeighbors(n_neighbors=n, metric=metric).fit(features)
  distances, indices = knn.kneighbors(features)
  return knn, distances, indices


def evaluate_knn_retrieval(features_compressed, class_ids, knn_model, k=5):
    precisions = []
    recalls = []

    for i in range(len(features_compressed)):
        query_label = class_ids[i]
        distances, indices = knn_model.kneighbors([features_compressed[i]], n_neighbors=k+1)

        retrieved_indices = indices[0][1:]
        retrieved_labels = class_ids[retrieved_indices]

        relevant = sum(retrieved_labels == query_label)
        precision = relevant / k

        total_relevant = np.sum(class_ids == query_label) - 1
        recall = relevant / total_relevant if total_relevant > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    return avg_precision, avg_recall


def plot_images_for_streamlit(filenames, distances):
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    fig = plt.figure(figsize=(12, 10))
    columns = 4
    for i, image in enumerate(images):
        ax = fig.add_subplot(int(len(images) / columns + 1), columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + extract_classname_filename(filenames[i]))
        else:
            ax.set_title("Similar Image\n" + extract_classname_filename(filenames[i]) + "\nDistance: " + f"{distances[i]:.2f}")
        ax.axis('off')
        ax.imshow(image)
    plt.tight_layout()
    return fig