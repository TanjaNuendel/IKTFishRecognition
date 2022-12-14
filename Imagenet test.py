#install dependencies
import os
import numpy as np
from keras.utils import load_img, img_to_array, image_dataset_from_directory
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from IPython.display import display, Image

old_model = InceptionV3(include_top=True, weights='imagenet')
new_model = load_model("model_with_aug2.h5")

def processing(img, size):
    image = load_img(img, target_size=(size, size))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

dir = "data"
def join(file):
    joined = os.path.join(dir, file).replace("\\","/")
    return joined

images = [join("fish_01/fish_000000589598_05327.png"), join("fish_02/fish_000010429596_03911.png"), join("fish_03/fish_000002199596_03708.png"),
          join("fish_07/aug_0_160.png"), join("fish_10/aug_0_273.png")]
def test_prediction(model, size, decode):
    for img in images:
        display(Image(filename = img, width = size, height = size))
        pred = model.predict(processing(img, size))
        if decode:
            print('Predicted:', decode_predictions(pred))
        else:
            print('Predicted:', pred)

test_prediction(old_model, 299,True)
test_prediction(new_model, 224, False)




