from cosine_similarity_layer import CosineDistance
from dataset_generator import SiameseDatasetGenerator
from model import SiameseModel, embedding
from tensorflow.keras import optimizers
import os


siamese_model = SiameseModel(embedding)

siamese_model.compile(optimizer=optimizers.Adam(0.0001))

left_images_path = os.path.join('C:/Users/Hazem/', 'left')
right_images_path = os.path.join('C:/Users/Hazem/', 'right')
img_shape = (245, 200, 3)


dataset = SiameseDatasetGenerator(left_images_path, right_images_path)

siamese_model.fit(dataset, epochs=2)



