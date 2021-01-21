from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend
from tensorflow.keras.preprocessing import image
from tensorflow import expand_dims
import random
import os
import numpy as np

class SiameseDatasetGenerator(Sequence):
    '''

    Dataset should have two directories,
    directory for anchor images and directory
    for positive image

    the negative images will be generated automatically

    anchor_images_path: the absolute path of the anchor images directory

    positive_images_path: the absolute path of the positive images directory

    image_shape: shape of the images in the dataset

    batch_size: how many examples per batch, default = 128

    shuffle: randomize batches when __getitem__ called


    e.g. dataset = SiameseDatasetGenerator(anchor_images_path, positive_images_path, image_shape)

    print(dataset[0]) -> will print a batch starting from index zero to (index+1) * batch_size

    each example has 3 images, anchor, positive and negative images

    '''
    def __init__(
        self,
        anchor_images_path,
        positive_images_path,
        image_shape,
        batch_size=128,
        shuffle=True,
    ):
        self.anchor_images_path = (
            anchor_images_path  # store the path of the anchor images
        )
        self.positive_images_path = (
            positive_images_path  # store the path of the positive images
        )
        self.image_shape = image_shape  ##store image shape
        # list the contents (images) of the specified directory
        self.anchor_images = os.listdir(positive_images_path)
        self.positive_images = os.listdir(positive_images_path)
        assert batch_size > 0, "Batch size should be greater than zero"
        self.batch_size = batch_size
        self.num_examples = len(self.anchor_images)
        self.num_batches = self.num_examples // batch_size
        self.shuffle = shuffle
        self.anchor_images = np.array(
            [
                os.path.join(self.anchor_images_path + "/", img)
                for img in self.anchor_images
            ]
        )
        self.positive_images = np.array(
            [
                os.path.join(self.positive_images_path + "/", img)
                for img in self.positive_images
            ]
        )


    """
    __len__ method is called to get the length of the batches
    it is called when we call len()
    """
    def __len__(self):
        return self.num_batches

    """
    this method allows us to get batches when we
    access the instance the same way we access a list
    e.g. dataset[0] will call __getitem__(index=0)
    """

    def __getitem__(self, index):
        # here we get batches of data by using slicing
        anchor_imgs = self.anchor_images[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        positive_imgs = self.positive_images[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        # store the loaded images to avoid reloading them in negative images
        # we store them in a set for faster access
        loaded_examples = set(anchor_imgs)

        # get negative_imgs by randomly choosing it from anchor or positive directory
        negative_imgs = [
            img
            for img in random.choice([self.anchor_images, self.positive_images])
            if img not in loaded_examples
        ]
        negative_imgs = np.array(random.choices(negative_imgs, k=self.batch_size))

        if self.shuffle:
            # create a list of random numbers to use it when we shuffle the batches
            random_shuffle = random.choices(
                [*range(self.batch_size)], k=self.batch_size
            )
            anchor_imgs = anchor_imgs[random_shuffle]
            positive_imgs = positive_imgs[random_shuffle]
            negative_imgs = negative_imgs[random_shuffle]

        anchor_imgs = self.preprocess_img(anchor_imgs)
        positive_imgs = self.preprocess_img(positive_imgs)
        negative_imgs = self.preprocess_img(negative_imgs)

        # here if the batch size equal one we just convert the images into numpy
        # and expand the dimension of this batch by adding 1 in the first axis
        if self.batch_size == 1:
            return expand_dims(
                np.array([anchor_imgs, positive_imgs, negative_imgs]), axis=0
            )
        # Add the batch_size dimension in the first axis by using permute()
        return backend.permute_dimensions(
            np.array([anchor_imgs, positive_imgs, negative_imgs]), (1, 0, 2, 3, 4)
        )

    def preprocess_img(self, imgs):
        # here we first load the images by using load_img()
        # then we convert them into tensors by using img_to_array()
        output = [image.img_to_array(image.load_img(img_path)) for img_path in imgs]
        if len(output) == 1:
            return output[0]
        return output