from tensorflow.keras import losses, layers


class CosineDistance(layers.Layer):
    def __init__(self):
        super(CosineDistance, self).__init__()
        
    def call(self, img1, img2):
        return -losses.CosineSimilarity(reduction=losses.Reduction.NONE)(img1, img2)


