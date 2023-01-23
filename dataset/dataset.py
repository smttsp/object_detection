import torch
import torch.nn as nn
import pandas as pd
import os
import PIL
# import skimage
# from skimage import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
# from tqdm import tqdm
import cv2
from collections import Counter


class ImageDataset(nn.Module):
    def __init__(self, file_dir, df_detection, df_segmentation=None):
        super().__init__()
        self.file_dir = file_dir
        self.df_detection = df_detection
        self.df_segmentation = df_segmentation

    def __getitem__(self, image_id):
        image = cv2.imread(os.path.join(self.file_dir, image_id + ".jpg"))
        objects = self.df_detection[self.df_detection["ImageID"] == image_id]

        bboxes = []
        for idx, object in objects.iterrows():
            xmin, xmax, ymin, ymax = object["XMin"], object["XMax"], object["YMin"], object["YMax"]
            category = object["LabelName"]
            is_group_of = object["IsGroupOf"]
            centerx = (xmin + xmax) / 2
            centery = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            bbox = [category, centerx, centery, width, height]
            bboxes.append(bbox)
        return image, bboxes
