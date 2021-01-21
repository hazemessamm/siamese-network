from tensorflow.keras import losses, layers

'''
Cosine Similarity Layer
This layer just computes how similar two feature vectors are by computing it using the Cosine Similarity We override the call method and implement our own call method.

Check out https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity

We return the negative of the loss because we just need to know how similar they are we do NOT need to know the loss

losses.reduction.NONE means that if we want the predictions as it is, we don't want to reduce the predictions by sum or by mean, this is useful of we want to make an inference on a batch of inputs

Check out https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction
'''

class CosineDistance(layers.Layer):
    def __init__(self):
        super(CosineDistance, self).__init__()
        
    def call(self, img1, img2):
        return -losses.CosineSimilarity(reduction=losses.Reduction.NONE)(img1, img2)


