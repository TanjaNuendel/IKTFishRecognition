from keras.layers import Dense, Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.utils import image_dataset_from_directory, split_dataset
from keras.models import Model
from glob import glob
import matplotlib.pyplot as plt

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

val_set, test_set = split_dataset(rem_set, left_size=0.5)

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in inception.layers:
    layer.trainable = False

folders = glob('data/*')

flat_layer = Flatten()(inception.output)

new_pred = Dense(len(folders), activation='softmax')(flat_layer)

model = Model(inputs=inception.input, outputs=new_pred)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

fitted_model = model.fit(train_set, validation_data=val_set, epochs=20, steps_per_epoch=len(train_set), validation_steps=len(val_set))
model.save('model_with_aug2.h5')

plt.plot(fitted_model.history['loss'], label='train loss')
plt.plot(fitted_model.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(fitted_model.history['accuracy'], label='train accuracy')
plt.plot(fitted_model.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()


