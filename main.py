import pandas
from dataset.dataset import ImageDataset

detection_file = "challenge-2019-validation-detection-bbox.csv"
dataset_dir = "/Users/samet/Overjet/datasets/object_detection_2019/validation/"

df_detection = pandas.read_csv(detection_file)
dataset = ImageDataset(dataset_dir, df_detection)

image_id = "000595fe6fee6369"
image, bboxes = dataset[image_id]
pass