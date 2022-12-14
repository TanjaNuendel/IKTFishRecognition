from keras.models import load_model
from keras.utils import image_dataset_from_directory, split_dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

IMAGE_SIZE = [224, 224]

directory = "data"
train_set, rem_set = image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="categorical",
    class_names=["fish_01", "fish_02", "fish_03", "fish_07", "fish_10"],
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="both",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

val_set, test_set = split_dataset(rem_set, left_size=0.8)

model = load_model("model_with_aug2.h5")
y_predict = np.argmax(model.predict(test_set), axis=1)
y_test = np.argmax(np.concatenate([y for x, y in test_set], axis=0), axis=1)

confusion = confusion_matrix(y_true=y_test, y_pred=y_predict, normalize="pred")
fish_names = ["fish_01", "fish_02", "fish_03", "fish_07", "fish_10"]
cmn = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=fish_names, yticklabels=fish_names)

plt.xlabel('Predictions')
plt.ylabel('True')
plt.show()




