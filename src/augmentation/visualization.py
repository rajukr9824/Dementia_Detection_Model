import matplotlib.pyplot as plt


def visualize_augmentation(
    original,
    augmented,
    augmentation_name="Augmented"
):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(augmented)
    plt.title(augmentation_name)
    plt.axis("off")

    plt.tight_layout()
    plt.show()