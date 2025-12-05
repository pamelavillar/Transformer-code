import torch
from torchvision import datasets, transforms
import numpy as np
import struct
import os

def save_mnist_test():
    """Guarda todas las imágenes TEST de MNIST en formato .bin"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Cargando MNIST test desde ./data ...")

    test_set = datasets.MNIST("./data", train=False, download=False, transform=transform)

    out_dir = "test_images"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Total imágenes de test: {len(test_set)}")
    print(f"Guardando en: {out_dir}/")

    for i in range(len(test_set)):
        image, label = test_set[i]
        image_np = image.numpy().flatten()  # (784,) float32 normalizada

        filename = f"{out_dir}/mnist_{i}_label_{label}.bin"
        with open(filename, "wb") as f:
            f.write(struct.pack(f"{len(image_np)}f", *image_np))

        if i % 2000 == 0:
            print(f"  Guardadas {i}/{len(test_set)}")

    print("\n====================================")
    print("   ✓ TEST MNIST generado correctamente")
    print(f"   Carpeta: {out_dir}/")
    print("====================================")


if __name__ == "__main__":
    save_mnist_test()
