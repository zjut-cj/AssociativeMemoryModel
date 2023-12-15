from typing import Optional, Callable, List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random


def normalize_word_vector(word, wordsList, wordVectors, mean_value, variance_value, left_boundary, right_boundary):

    word_index = wordsList.index(word)
    word_embedding = wordVectors[word_index]

    embedding_n01 = (word_embedding - np.array([mean_value] * 100)) / np.array([np.sqrt(variance_value)] * 100)
    embedding_norm = np.array([0] * 100, dtype=np.float32)

    for k in range(100):
        if word_embedding[k] < left_boundary:
            embedding_norm[k] = -3
        elif word_embedding[k] > right_boundary:
            embedding_norm[k] = 3
        else:
            embedding_norm[k] = embedding_n01[k]

    embedding_norm = (embedding_norm + np.array([np.abs(3)] * 100)) / (3 * 2)
    embedding_norm = np.clip(embedding_norm, a_min=0, a_max=1)

    return embedding_norm


class FashionMNISTDataset(Dataset):
    def __init__(self, root: str, train: bool, classes: int, dataset_size: int,
                 image_transform: Optional[Callable] = None) -> None:
        self.classes = classes
        self.dataset_size = dataset_size
        self.image_transform = image_transform
        self.fashion_mnist_data = datasets.FashionMNIST(root=root, train=train, transform=image_transform, download=True)
        # Organize data by class
        self.data_by_class = {}
        for i in range(10):
            self.data_by_class[i] = []

        for index, (image, label) in enumerate(self.fashion_mnist_data):
            self.data_by_class[label].append((image, label, index))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index) -> Tuple:

        # Get 10 random images and their labels (one per class)
        random_images = []
        random_text = []
        random_labels = []
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                       'ankle', 'boot']

        wordsList = np.load('/home/jww/storage/glove.6B/wordsList_100d.npy')

        wordsList = wordsList.tolist()  # Originally loaded as numpy array
        wordVectors = np.load('/home/jww/storage/glove.6B/wordVectors_100d.npy')

        mean_value = np.mean(wordVectors)
        variance_value = np.var(wordVectors)
        left_boundary = mean_value - 3 * np.sqrt(variance_value)
        right_boundary = mean_value + 3 * np.sqrt(variance_value)
        for i in range(self.classes):
            images_text = []
            # images_text.append(normalize_word_vector('a', wordsList, wordVectors, mean_value, variance_value,
            #                                          left_boundary, right_boundary))
            # images_text.append(normalize_word_vector('photo', wordsList, wordVectors, mean_value, variance_value,
            #                                          left_boundary, right_boundary))
            # images_text.append(normalize_word_vector('of', wordsList, wordVectors, mean_value, variance_value,
            #                                          left_boundary, right_boundary))
            class_data = self.data_by_class[i]
            random_image, random_label, _ = random.choice(class_data)
            random_images.append(random_image)
            random_labels.append(random_label)
            if random_label == 9:
                images_text.append(normalize_word_vector(text_labels[9], wordsList, wordVectors, mean_value,
                                                         variance_value, left_boundary, right_boundary))
                images_text.append(
                    normalize_word_vector(text_labels[10], wordsList, wordVectors, mean_value, variance_value,
                                          left_boundary, right_boundary))
            else:
                images_text.append(normalize_word_vector('a', wordsList, wordVectors, mean_value, variance_value,
                                                         left_boundary, right_boundary))
                images_text.append(
                    normalize_word_vector(text_labels[random_label], wordsList, wordVectors, mean_value, variance_value,
                                          left_boundary, right_boundary))
            random_text.append(torch.Tensor(np.array(images_text, dtype=np.float32)))
        # Stack and concatenate to form the desired shape
        random_images = torch.stack(random_images, dim=0)
        random_text = torch.stack(random_text, dim=0)
        random_labels = torch.tensor(random_labels)
        # Get a random image and its label
        random_image_index = random.choice(range(self.classes))
        random_image = random_images[random_image_index]
        random_image_text = random_text[random_image_index]
        random_image_label = random_labels[random_image_index]

        return (
            random_images,  # List of 10 random images
            random_text,  # List of 10 corresponding labels
            random_labels,
            random_image,   # Random image from the 10
            random_image_text,  # Label of the random image
            random_image_label
        )


class SequentialFashionMNISTDataset(Dataset):
    def __init__(self, root: str, train: bool, classes: int, dataset_size: int,
                 image_transform: Optional[Callable] = None) -> None:
        self.classes = classes
        self.dataset_size = dataset_size
        self.image_transform = image_transform
        self.fashion_mnist_data = datasets.FashionMNIST(root=root, train=train, transform=image_transform, download=True)
        # Organize data by class
        self.data_by_class = {}
        for i in range(10):
            self.data_by_class[i] = []

        for index, (image, label) in enumerate(self.fashion_mnist_data):
            self.data_by_class[label].append((image, label, index))

        # Initialize indices for each class
        self.class_indices = [0] * classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index) -> Tuple:

        # Get 10 random images and their labels (one per class)
        random_images = []
        random_text = []
        random_labels = []
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                       'ankle', 'boot']

        wordsList = np.load('/home/jww/storage/glove.6B/wordsList_100d.npy')

        wordsList = wordsList.tolist()  # Originally loaded as numpy array
        wordVectors = np.load('/home/jww/storage/glove.6B/wordVectors_100d.npy')

        mean_value = np.mean(wordVectors)
        variance_value = np.var(wordVectors)
        left_boundary = mean_value - 3 * np.sqrt(variance_value)
        right_boundary = mean_value + 3 * np.sqrt(variance_value)
        for i in range(self.classes):
            images_text = []
            images_text.append(normalize_word_vector('a', wordsList, wordVectors, mean_value, variance_value,
                                                     left_boundary, right_boundary))
            images_text.append(normalize_word_vector('photo', wordsList, wordVectors, mean_value, variance_value,
                                                     left_boundary, right_boundary))
            images_text.append(normalize_word_vector('of', wordsList, wordVectors, mean_value, variance_value,
                                                     left_boundary, right_boundary))
            class_data = self.data_by_class[i]
            # Use the class index to get the next item in class_data
            random_image, random_label, _ = class_data[self.class_indices[i]]
            random_images.append(random_image)
            random_labels.append(random_label)
            if random_label == 9:
                images_text.append(normalize_word_vector(text_labels[9], wordsList, wordVectors, mean_value,
                                                         variance_value, left_boundary, right_boundary))
                images_text.append(
                    normalize_word_vector(text_labels[10], wordsList, wordVectors, mean_value, variance_value,
                                          left_boundary, right_boundary))
            else:
                images_text.append(normalize_word_vector('a', wordsList, wordVectors, mean_value, variance_value,
                                                         left_boundary, right_boundary))
                images_text.append(
                    normalize_word_vector(text_labels[random_label], wordsList, wordVectors, mean_value, variance_value,
                                          left_boundary, right_boundary))
            random_text.append(torch.Tensor(np.array(images_text, dtype=np.float32)))
            # Update the class index for the next item
            self.class_indices[i] = (self.class_indices[i] + 1) % len(class_data)

        # Stack and concatenate to form the desired shape
        random_images = torch.stack(random_images, dim=0)
        random_text = torch.stack(random_text, dim=0)
        random_labels = torch.tensor(random_labels)
        # Get a random image and its label
        random_image_index = random.choice(range(self.classes))
        random_image = random_images[random_image_index]
        random_image_text = random_text[random_image_index]
        random_image_label = random_labels[random_image_index]

        return (
            random_images,  # List of 10 random images
            random_text,  # List of 10 corresponding labels
            random_labels,
            random_image,  # Random image from the 10
            random_image_text,  # Label of the random image
            random_image_label
        )


class HeteroAssociativeFashionMNISTDataset(Dataset):
    def __init__(self, root: str, train: bool, classes: int, dataset_size: int,
                 image_transform: Optional[Callable] = None) -> None:
        self.classes = classes
        self.dataset_size = dataset_size
        self.image_transform = image_transform
        self.fashion_mnist_data = datasets.FashionMNIST(root=root, train=train, transform=image_transform, download=True)
        # Organize data by class
        self.data_by_class = {}
        for i in range(10):
            self.data_by_class[i] = []

        for index, (image, label) in enumerate(self.fashion_mnist_data):
            self.data_by_class[label].append((image, label, index))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index) -> Tuple:
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
        random_image_data = self.data_by_class[random_image_index]
        random_image, random_image_label, _ = random.choice(random_image_data)
        # random_image = random_images[random_image_index]
        # random_image_label = random_labels[random_image_index]

        return (
            random_images,  # List of 10 random images
            random_labels,  # List of 10 corresponding labels
            random_image,   # Random image from the 10
            random_image_label  # Label of the random image
        )


class SameCategoryFashionMNISTDataset(Dataset):
    def __init__(self, root: str, train: bool, classes: int, dataset_size: int,
                 image_transform: Optional[Callable] = None) -> None:
        self.classes = classes
        self.dataset_size = dataset_size
        self.image_transform = image_transform
        self.fashion_mnist_data = datasets.FashionMNIST(root=root, train=train, transform=image_transform, download=True)
        # Organize data by class
        self.data_by_class = {}
        for i in range(10):
            self.data_by_class[i] = []

        for index, (image, label) in enumerate(self.fashion_mnist_data):
            self.data_by_class[label].append((image, label, index))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index) -> Tuple:

        # Get 10 random images and their labels (one per class)
        random_images = []
        random_labels = []
        for i in range(self.classes):
            class_data = self.data_by_class[8]
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

    fashion_mnist_dataset = SequentialFashionMNISTDataset(root='/usr/common/datasets/FashionMNIST', train=True,
                                                classes=10, dataset_size=100, image_transform=transform)
    image_sequence, text, labels, image_query, targets, target_label = fashion_mnist_dataset[0]

    dataloader = DataLoader(fashion_mnist_dataset, batch_size=128, shuffle=True)
    for batch in dataloader:
        random_images, random_text, random_labels, random_image, random_image_text, random_image_label = batch
        # Now you have the required data for each batch
        random_images_array = random_images.clone().detach().numpy()
        random_text_array = random_text.clone().detach().numpy()
        random_labels_array = random_labels.clone().detach().numpy()
        random_image_array = random_image.clone().detach().numpy()
        random_image_text_array = random_image_text.clone().detach().numpy()
        random_image_label_array = random_image_label.clone().detach().numpy()
        print(random_images, random_labels, random_image, random_image_label)
