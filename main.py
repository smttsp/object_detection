import pandas
from dataset.dataset import ImageDataset

detection_file = "challenge-2019-validation-detection-bbox.csv"


df_detection = pandas.read_csv(detection_file)
dataset = ImageDataset("", df_detection)

image_id = "000595fe6fee6369"
x = dataset[image_id]
pass