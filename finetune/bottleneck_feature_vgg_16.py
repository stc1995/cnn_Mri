from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'old_0.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
print(features.shape)
features = [features]
for i in range(98):
    img_path = 'old_%d.jpg' % i
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    temp = model.predict(x)
    features.append(temp)

features = np.array(features)
print(features.shape) (99, 1, 7, 7, 512)
np.save('old_features.npy', features)


img_path = 'ad_0.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
print(features.shape)
features = [features]
for i in range(100):
    img_path = 'ad_%d.jpg' % i
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    temp = model.predict(x)
    features.append(temp)

features = np.array(features)
print(features.shape)       #(101, 1, 7, 7, 512)
np.save('ad_features.npy', features)
