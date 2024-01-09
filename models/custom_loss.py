import tensorflow
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model


# Load VGG16 without the classification head, and weights use imagenet
vgg16 = VGG16(include_top=False, weights='imagenet')
vgg16.trainable = False

# Use features from a shallower layer
layer_name = 'block2_conv2'
vgg_layer_output = vgg16.get_layer(layer_name).output

# Feature extraction model
feature_extractor = Model(inputs=vgg16.input, outputs=vgg_layer_output)

def simplified_perceptual_loss(y_true, y_pred):
    # Preprocess input for VGG16
    y_true_processed = tensorflow.keras.applications.vgg16.preprocess_input(y_true)
    y_pred_processed = tensorflow.keras.applications.vgg16.preprocess_input(y_pred)
    # Extract features
    true_features = feature_extractor(y_true_processed)
    pred_features = feature_extractor(y_pred_processed)
    # Compute loss as mean squared error between feature representations
    return tensorflow.reduce_mean(tensorflow.square(pred_features - true_features))
