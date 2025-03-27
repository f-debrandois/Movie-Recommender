from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, labels, img_path, transform=True):

        self.labels = labels
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):

        img_path = self.img_path[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, img_path, label