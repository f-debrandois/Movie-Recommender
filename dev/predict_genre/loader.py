import sys, os

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
from sklearn.model_selection import train_test_split




class CustomImageDataset(Dataset):  
    def __init__(self, labels, img_path, transform=True):

        self.labels = labels
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):

        img_path = self.img_path[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, img_path, label




def get_labels(registers, data_path):

    train_paths, test_paths = [], []
    train_labels, test_labels = [], []

    for (i, register) in enumerate(registers):

        register_path = os.path.join(data_path, register)
        register_images = [os.path.join(register_path, file) for file in os.listdir(register_path)]
        train_img, test_img = train_test_split(register_images, test_size=0.2)
    
        train_paths += train_img
        test_paths += test_img
        train_labels += [i] * len(train_img)
        test_labels += [i] * len(test_img)

    return train_labels, test_labels, train_paths, test_paths




def generate_loaders(registers, data_path):

    initial_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

    train_labels, test_labels, train_paths, test_paths = get_labels(registers, data_path)

    train_dataset = CustomImageDataset(labels=train_labels, img_path=train_paths, transform=initial_transform)
    test_dataset = CustomImageDataset(labels=test_labels, img_path=test_paths, transform=initial_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader