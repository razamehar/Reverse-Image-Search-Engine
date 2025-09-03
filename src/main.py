'''import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import *
from config.constants import N


img_path = input("Provide image: ")


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_dir = os.path.join(project_root, "features_data")

features = np.load(os.path.join(features_dir, "features.npy"))
filenames = np.load(os.path.join(features_dir, "filenames.npy"), allow_pickle=True)
class_ids = np.load(os.path.join(features_dir, "class_ids.npy"))


_, feature_extractor = define_model()

knn_euc, distances, indices = apply_knn(features, n=N, metric='euclidean')

def extract_custom_image_feature(img_path):
    return extract_features(img_path, feature_extractor)


custom_feature = extract_custom_image_feature(img_path).reshape(1, -1)

distances, indices = knn_euc.kneighbors(custom_feature)

similar_image_paths = [img_path] + [filenames[i] for i in indices[0][1:4]]

plot_images(similar_image_paths, distances[0])'''
