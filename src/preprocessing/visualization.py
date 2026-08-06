import matplotlib.pyplot as plt

def visualize_preprocessing(original, resized, clahe_image):

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(resized)
    plt.title("Resized")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(clahe_image)
    plt.title("CLAHE")
    plt.axis("off")

    plt.tight_layout()
    plt.show()