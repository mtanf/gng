import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DynamicUpsample(tf.keras.layers.Layer):
    def __init__(self, method='bilinear', **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def call(self, inputs, ref_tensor):
        return tf.image.resize(inputs, (tf.shape(ref_tensor)[1], tf.shape(ref_tensor)[2]), method=self.method)

def DeeplabV3Plus(num_classes,
                  filters_conv1=24, filters_conv2=24,
                  filters_spp=128, filters_final=128,
                  dilated_conv_rates =[1, 4, 8, 16],
                  trainable_resnet=True):

    #Defining inputs
    model_input = keras.Input(shape=(None, None, 3))
    #Preprocessing layer
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=preprocessed)
    resnet50.trainable=trainable_resnet
    # mobilenet = keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_tensor=preprocessed)
    # mobilenet.trainable=trainable_resnet

    #Feature extraction
    x = resnet50.get_layer("conv4_block6_2_relu").output #Layer containing high-level feature representations
    input_b = resnet50.get_layer("conv2_block3_2_relu").output #Layer containing features at an earlier stage of the network's processing (lower level)


    '''
    The next section of the network applies global average pooling to obtain global context information,
    followed by dimensionality reduction through convolution and normalization.
    The dynamically upsampled features are then combined with the original feature map to enrich the feature representation.
    '''
    #Pooling operation to collapse the spatial dimensions of each feature map into a single value
    x1 = layers.GlobalAveragePooling2D()(x)
    x1 = layers.Reshape((1, 1, x.shape[-1]))(x1)
    #Projecting the feature maps into a lower-dimensional space (of size filters_conv1). 
    x1 = layers.Conv2D(filters=filters_conv1, kernel_size=1, padding="same")(x1) #Used to learn better features and reduce the computational cost.
    x1 = layers.BatchNormalization()(x1)
    #Upsample to x size
    x1 = DynamicUpsample()(x1, x)

    
    '''
    This section of the code implements Atrous Spatial Pyramid Pooling
    '''
    #Modification: as the rate increases, the kernel sizes increase too 
    pyramids = []
    for rate in dilated_conv_rates:
        if rate == 1:
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3, dilation_rate=rate, padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)
        else:
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3 + int(rate*(1/2)), dilation_rate=rate, padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)

    x = layers.Concatenate(axis=-1)([x1] + pyramids)
    #Convolution to reduce the dimensionality and computational cost
    x = layers.Conv2D(filters=filters_spp, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    #Adjusting the dimensions of lower level representations using convolutional layer
    input_b = layers.Conv2D(filters=filters_conv2, kernel_size=1, padding="same")(input_b)
    input_b = layers.BatchNormalization()(input_b)
    #Upsample to x size
    input_a = DynamicUpsample()(x, input_b)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    #These convolutional layers refine the combined features before the final classification
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    #Final upsampling ensures that the final output matches the spatial dimensions of the input image
    x = DynamicUpsample()(x, model_input)
    #Classification layer
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model = keras.Model(inputs=model_input, outputs=model_output)
    return model


def DeeplabV3Plus_mobilenet(num_classes,
                  filters_conv1=24, filters_conv2=24,
                  filters_spp=128, filters_final=128,
                  dilated_conv_rates=[1, 4, 8, 16],
                  trainable_mobilenet=True):

    #Defining inputs
    model_input = keras.Input(shape=(None, None, 3))
    #Preprocessing layer
    preprocessed = keras.applications.mobilenet_v3.preprocess_input(model_input)
    mobilenetv3 = keras.applications.MobileNetV3Large(weights="imagenet", include_top=False, input_tensor=preprocessed)
    mobilenetv3.trainable = trainable_mobilenet

    #Feature extraction
    x = mobilenetv3.get_layer("expanded_conv_12/project").output #Layer containing high-level feature representations
    input_b = mobilenetv3.get_layer("expanded_conv_3/project").output #Layer containing features at an earlier stage of the network's processing (lower level)

    '''
    The next section of the network applies global average pooling to obtain global context information,
    followed by dimensionality reduction through convolution and normalization.
    The dynamically upsampled features are then combined with the original feature map to enrich the feature representation.
    '''
    #Pooling operation to collapse the spatial dimensions of each feature map into a single value
    x1 = layers.GlobalAveragePooling2D()(x)
    x1 = layers.Reshape((1, 1, x.shape[-1]))(x1)
    #Projecting the feature maps into a lower-dimensional space (of size filters_conv1). 
    x1 = layers.Conv2D(filters=filters_conv1, kernel_size=1, padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    #Upsample to x size
    x1 = DynamicUpsample()(x1, x)

    '''
    This section of the code implements Atrous Spatial Pyramid Pooling
    '''
    #Modification: as the rate increases, the kernel sizes increase too 
    pyramids = []
    for rate in dilated_conv_rates:
        if rate == 1:
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3, dilation_rate=rate, padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)
        else:
            pyramid = layers.Conv2D(filters=filters_spp, kernel_size=3 + int(rate*(1/2)), dilation_rate=rate, padding="same")(x)
            pyramid = layers.BatchNormalization()(pyramid)
            pyramids.append(pyramid)

    x = layers.Concatenate(axis=-1)([x1] + pyramids)
    #Convolution to reduce the dimensionality and computational cost
    x = layers.Conv2D(filters=filters_spp, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    #Adjusting the dimensions of lower level representations using convolutional layer
    input_b = layers.Conv2D(filters=filters_conv2, kernel_size=1, padding="same")(input_b)
    input_b = layers.BatchNormalization()(input_b)
    #Upsample to x size
    input_a = DynamicUpsample()(x, input_b)

    #These convolutional layers refine the combined features before the final classification
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=filters_final, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    #Final upsampling ensures that the final output matches the spatial dimensions of the input image
    x = DynamicUpsample()(x, model_input)
    #Classification layer
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model = keras.Model(inputs=model_input, outputs=model_output)
    return model