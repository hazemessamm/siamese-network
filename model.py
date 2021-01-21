from tensorflow.keras import applications
from tensorflow.keras import Model, layers
import tensorflow as tf


img_shape=(254, 200, 3)

#Loading a pre-trained model
#Here we use ResNet50 architecture, we use "imagenet" weights,
#also we pass the image shape Note that include_top means that we do NOT want the top layers
base_cnn = applications.ResNet50(weights='imagenet', input_shape=img_shape, include_top=False)


#Fine Tuning
#Here we fine tune the ResNet50 we freeze all layers that exist before "conv5_block1_out" layer, 
#starting from "conv5_block2_2_relu" layer we unfreeze all the layers so we can just train these layers
trainable = False
for layer in base_cnn.layers:
    if layer.name == 'conv5_block1_out':
        trainable = True
    layer.trainable = trainable


#Adding top layers
#Here we customize the model by adding Dense layers and Batch Normalization layers. 
# we start with the image input then we pass the input to the base_cnn then we flatten it. 
#Finally we pass each layer as an input to the next layer 
#the output layer is just a dense layer which will act as an embedding for our images.
flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation='relu')(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation='relu')(dense1)
dense2 = layers.Dropout(0.25)(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name='SiameseNetwork')



#Model for training
#This model is just used for training we pass to it three input batches (anchor images, positive images, 
#negative images) and the output will be the output of the model we defined above, 
#it will be 1 output for each input.
anchor_input = layers.Input(shape=img_shape)
positive_input = layers.Input(shape=img_shape)
negative_input = layers.Input(shape=img_shape)

anchor_output = embedding(anchor_input)
positive_output = embedding(positive_input)
negative_output = embedding(negative_input)

training_model = Model([anchor_input, positive_input, negative_input], 
                       {"anchor_embedding": anchor_output, 
                        "positive_embedding": positive_output, "negative_embedding":negative_output})



'''
Model subclassing

Here we customize our training process and our model.

We override the train_step() method and apply our own loss and our own training process

We also use Triplet loss function as we specified above.

Loss function explaination:

we calculate the distance between the anchor embedding and the positive embedding the axis = -1 
because we want the distance over the features of every example. We also add alpha which act as extra margin.
'''
class SiameseModel(Model):
    def __init__(self, model, alpha=0.5):
        super(SiameseModel, self).__init__()
        self.embedding = model #we pass the model to the class
        self.alpha = alpha
        
    def call(self, inputs):
        pass
        
    def train_step(self, data):
        #here we create a tape to record our operations so we can get the gradients
        with tf.GradientTape() as tape:
            embeddings = training_model((data[:, 0], data[:, 1], data[:, 2]))
            
            #Euclidean Distance between anchor and positive
            #axis=-1 so we can get distances over examples
            anchor_positive_dist = tf.reduce_sum(
                tf.square(embeddings['anchor_embedding'] - embeddings['positive_embedding']), -1)
            
            #Euclidean Distance between anchor and negative
            anchor_negative_dist = tf.reduce_sum(
                tf.square(embeddings['anchor_embedding'] - embeddings['negative_embedding']), -1)
            
            #getting the loss by subtracting the distances
            loss = anchor_positive_dist - anchor_negative_dist
            #getting the max because we don't want negative loss
            loss = tf.reduce_sum(tf.maximum(loss+self.alpha, 0.0))
        #getting the gradients [loss with respect to trainable weights]
        grads = tape.gradient(loss, training_model.trainable_weights)
        #applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(zip(grads, training_model.trainable_weights))
        return {"Loss": loss}
