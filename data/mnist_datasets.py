from typing import Optional, Callable, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random


class MNISTDataset(Dataset):
    def __init__(self, root: str, train: bool, classes: int, dataset_size: int,
                 image_transform: Optional[Callable] = None) -> None:
        self.classes = classes
        self.dataset_size = dataset_size
        self.image_transform = image_transform
        self.mnist_data = datasets.MNIST(root=root, train=train, transform=image_transform, download=True)
        # Organize data by class
        self.data_by_class = {}
        for i in range(10):
            self.data_by_class[i] = []

        for index, (image, label) in enumerate(self.mnist_data):
            self.data_by_class[label].append((image, label, index))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index) -> Tuple:
        # Get the image and label at the given index
        image, label = self.mnist_data[index]

        # Get 10 random images and their labels (one per class)
        random_images = []
        random_labels = []
        for i in range(self.classes):
            class_data = self.data_by_class[i]
            random_image, random_label, _ = random.choice(class_data)
            random_images.append(random_image)
            random_labels.append(random_label)

        # Stack and concatenate to form the desired shape
        random_images = torch.stack(random_images, dim=0)
        random_labels = torch.tensor(random_labels)
        # Get a random image and its label
        random_image_index = random.choice(range(self.classes))
        random_image = random_images[random_image_index]
        random_image_label = random_labels[random_image_index]

        return (
            random_images,  # List of 10 random images
            random_labels,  # List of 10 corresponding labels
            random_image,   # Random image from the 10
            random_image_label  # Label of the random image
        )


# Example usage:
if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mnist_dataset = MNISTDataset(root='/usr/common/datasets/MNIST', train=True, classes=10, dataset_size=6000, image_transform=transform)
    image_sequence, labels, image_query, targets = mnist_dataset[0]

    dataloader = DataLoader(mnist_dataset, batch_size=128, shuffle=True)
    for batch in dataloader:
        random_images, random_labels, random_image, random_image_label = batch
        # Now you have the required data for each batch
        print(random_images, random_labels, random_image, random_image_label)
