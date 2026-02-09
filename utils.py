import matplotlib.pyplot as plt
import numpy as np


def plot_example(dataset, num):

    plt.figure(figsize=(4,2))

    image, label = dataset[num]

    img_display = image.squeeze().numpy()
    plt.imshow(img_display, cmap= 'gray')
    plt.title(f"Label: Example {num}")
    plt.axis('off')
    plt.show()


def plot_example2(img):

    plt.figure(figsize=(4,2))

    img_display = img.numpy()
    plt.imshow(img_display, cmap= 'gray')
    plt.title("Example")
    plt.axis('off')
    plt.show()


def plot_example3(img1, img2):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))

    im1 = ax1.imshow(img1, cmap='viridis')
    ax1.set_title("Image 1")
    im2 = ax2.imshow(img1, cmap='magma')
    ax2.set_title("Image 2")

    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def heatmap(data, title):

    plt.imshow(data, cmap="hot_r", interpolation="nearest")
    plt.colorbar(label= 'Intensity')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()    


def plot_loss(loss_data):
    plt.plot(loss_data)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss(Cosine)")
    plt.title("Training Loss over Time")
    plt.show()

def multi_plot(data1, data2):

    iter = range(1, len(data1)+1)
    plt.plot(iter, data1, label= "APS")
    plt.plot(iter, data2, label= "Embeddings")
    plt.xlabel("Iteration")
    plt.ylabel("Cosine Similarity")
    plt.title("Similarity APS - Embeddings")
    plt.legend()
    plt.show()


# dt = np.array([
#     [1,1,1,1],
#     [2,2,2,4],
#     [5,5,5,8]
# ])

# heatmap(dt)