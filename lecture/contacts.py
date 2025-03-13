"""Out of scope for Tripos."""

import os
import matplotlib.pyplot as plt


def main(portrait=False, dpi=75):
    if portrait:
        fig = plt.figure(figsize=(4, 10))
    else:
        fig = plt.figure(figsize=(10, 4))

    images = []
    for name in os.listdir("contacts"):
        if name.endswith(".png"):
            images.append((name[:-4], plt.imread(os.path.join("contacts", name))))

    for i, (title, image) in enumerate(images):
        if portrait:
            ax = fig.add_subplot(5, 2, i + 1)
        else:
            ax = fig.add_subplot(2, 5, i + 1)

        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig("contacts.png", dpi=dpi)


if __name__ == "__main__":
    main(False, 300)
