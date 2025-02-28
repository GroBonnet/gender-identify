import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def show_pictures(path_images, batch_size=10, color='RVB'):
    
    valid_extensions = (".npy", ".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(path_images) if f.lower().endswith(valid_extensions)]

    batch_im = files[:min(batch_size, len(files))]

    cols = 5
    rows = (len(batch_im) // cols) + (1 if len(batch_im) % cols else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))

    axes = axes.flatten() if rows > 1 else np.array([axes]).flatten()

    for ax, image in zip(axes, batch_im):
        image_path = os.path.join(path_images, image)
        
        if color == 'gray' and image.endswith(".npy"):
            img = np.load(image_path)
            ax.imshow(img, cmap='gray')
        else:
            img = Image.open(image_path)
            ax.imshow(img)

        ax.axis("off")

    for ax in axes[len(batch_im):]:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs("output", exist_ok=True) 
    plt.savefig("output/output.png")
