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

#here we just load from the dataset an example
#we should NOT test the performace of the model 
#using training data but here we are just see how did it learn
example_prediction = dataset[9]
anchor_example = image.array_to_img(example_prediction[:, 0][0])
positive_example = image.array_to_img(example_prediction[:, 1][0])
negative_example = image.array_to_img(example_prediction[:, 2][0])

#we add an extra dimension (batch_size dimension) in the first axis by using expand dims.
anchor_tensor = np.expand_dims(example_prediction[:, 0][0], axis=0)
positive_tensor = np.expand_dims(example_prediction[:, 1][0], axis=0)
negative_tensor = np.expand_dims(example_prediction[:, 2][0], axis=0)

anchor_embedding, positive_embedding = embedding(anchor_tensor), embedding(positive_tensor)
positive_similarity = CosineDistance()(anchor_embedding, positive_embedding)
print("Similarity:", positive_similarity)

anchor_embedding, negative_embedding = embedding(anchor_tensor), embedding(negative_tensor)
negative_similarity = CosineDistance()(anchor_embedding, negative_embedding)
print("Similarity:", negative_similarity)
