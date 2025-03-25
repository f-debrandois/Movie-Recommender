import torch
import torchvision
import pandas as pd
import random
from torchvision import transforms
from PIL import Image


# DÉFINITION DES TRANSFORMATIONS
# ON VEUT :
# - REDIMENSIONNER LES IMAGES EN 185x185 (PROVIENT DU `MIN_SIZE` TROUVÉ DANS LE NOTEBOOK)
# - CONVERTIR LES IMAGES EN NIVEAUX DE GRIS (UTILITE A VERIFIER)
# - APPLIQUER UN MASQUE AVEC UNE CERTAINE PROBABILITÉ (À VÉRIFIER SI CELA EST UTILE/PERTINENT, DES TESTS POURRAIENT ÊTRE FAITS)
class TrainTransform:
    def __init__(self, resize=(185, 185), mean=(0.5,), std=(0.5,), mask_prob=0.5, mask_size=32):
        self.resize = transforms.Resize(resize)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.mask_prob = mask_prob
        self.mask_size = mask_size

    def __call__(self, img):
        img = self.resize(img)
        img = self.to_tensor(img)

        # APPLICATION D'UN MASQUE AVEC PROBABILITÉ `mask_prob` DE TAILLE `mask_size`, TIRÉ ALÉATOIREMENT SUR L'IMAGE
        if random.random() < self.mask_prob:
            img = self.apply_random_mask(img)

        img = self.normalize(img)
        return img

    def apply_random_mask(self, img):
        _, h, w = img.shape  # img est de forme (C, H, W)
        mask_size = min(self.mask_size, h, w)  # on s'assure que le masque ne dépasse pas la taille de l'image

        x = random.randint(0, w - mask_size)
        y = random.randint(0, h - mask_size)

        img[:, y:y+mask_size, x:x+mask_size] = 0  # Masquage en noir (peut être ajusté)
        return img
    



class TestTransform:
    def __init__(self, resize=(185, 185), mean=(0.5,), std=(0.5,), mask_prob=0.5, mask_size=32):
        self.resize = transforms.Resize(resize)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.mask_prob = mask_prob
        self.mask_size = mask_size

    def __call__(self, img):
        img = self.resize(img)
        img = self.to_tensor(img)

        img = self.normalize(img)
        return img
    




class TrainTransformWithGrayscale:
    def __init__(self, resize=(185, 185), mean=(0.5,), std=(0.5,), mask_prob=0.5, mask_size=32):
        self.resize = transforms.Resize(resize)
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)  # Convertir en niveaux de gris
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.mask_prob = mask_prob
        self.mask_size = mask_size

    def __call__(self, img):
        img = self.resize(img)
        img = self.to_grayscale(img)
        img = self.to_tensor(img)

        # APPLICATION D'UN MASQUE AVEC PROBABILITÉ `mask_prob` DE TAILLE `mask_size`, TIRÉ ALÉATOIREMENT SUR L'IMAGE
        if random.random() < self.mask_prob:
            img = self.apply_random_mask(img)

        img = self.normalize(img)
        return img

    def apply_random_mask(self, img):
        _, h, w = img.shape  # img est de forme (C, H, W)
        mask_size = min(self.mask_size, h, w)  # on s'assure que le masque ne dépasse pas la taille de l'image

        x = random.randint(0, w - mask_size)
        y = random.randint(0, h - mask_size)

        img[:, y:y+mask_size, x:x+mask_size] = 0  # Masquage en noir (peut être ajusté)
        return img
    



class TestTransformWithGrayscale:
    def __init__(self, resize=(185, 185), mean=(0.5,), std=(0.5,), mask_prob=0.5, mask_size=32):
        self.resize = transforms.Resize(resize)
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)  # Convertir en niveaux de gris
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.mask_prob = mask_prob
        self.mask_size = mask_size

    def __call__(self, img):
        img = self.resize(img)
        img = self.to_grayscale(img)
        img = self.to_tensor(img)

        img = self.normalize(img)
        return img