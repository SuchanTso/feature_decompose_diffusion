import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

# Load the RGB image for PCA
image_path = "/data/usr/zsc/project/feature_decompose_diffusion/datasets/lsun_bedroom/00000089629ce3ba87bae003073896ba01988dee.jpg"
image = Image.open(image_path)
image_rgb = np.array(image)  # Convert to numpy array
h, w, c = image_rgb.shape
flat_image_rgb = image_rgb.reshape(-1, c)

# Perform PCA with 3 components
pca_3 = PCA(n_components=3)
pca_result_3 = pca_3.fit_transform(flat_image_rgb)

# Reconstruct the PCA components and normalize for visualization
components_images = [pca_result_3[:, i].reshape(h, w) for i in range(3)]
components_images_normalized = [
    (comp - np.min(comp)) / (np.max(comp) - np.min(comp)) for comp in components_images
]

# Visualize the first 3 PCA components in separate grayscale plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(components_images_normalized[i], cmap='gray')
    ax.set_title(f"PCA Component {i + 1}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("./test.jpg")