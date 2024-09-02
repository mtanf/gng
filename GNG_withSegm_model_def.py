#GNG-Classifier model definition 
#Assuming: input RGB of MTCNN face + DeepLabV3Plus face parts segmentation

####### Workflow #######
# Original RGB image goes into a mobilenet (imagenet weights) encoder

# Input must be a single tensor since SHAP will need to work on it like a single object

import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input,GlobalAveragePooling2D, Conv2D, Flatten, Concatenate
from keras.preprocessing.image import ImageDataGenerator

#Creating GNG classifier with face segmentation ( (MobilenetV3Large encoder) || (DeepLabV3Plus for face segmentation + ConvNet for segmentation encoding)  + MLP classifier on top applied to RGB images for fake vs réal classification)
class gng_seg:
    """
    A class for building a GNG classifier model that combines a MobileNetV3Large encoder for RGB images and a custom
    MLP classifier for face authenticity classification. It also includes a separate encoder for face parts
    segmentation input.
    """

    def __init__(self, input_shape, mlp_units, trained_segmentator_mdl, seg_conv_params,
                  enc_output_layer_activ="relu", encoding_dim=None, mlp_head_activ="relu", mlp_output_layer_activ="sigmoid",
                   seg_encoding_dim=None, seg_padding="same", seg_enc_activation="relu", encoding_with_same_size = False,
                    optimizer='adam', learning_rate=0.001, loss='mse'):
        """
        Initializes the gng_basic model

        Args:
            input_shape (tuple): The shape of the input data, including channels (e.g., (224, 224, 3) for RGB images).
            mlp_units (list): A list specifying the number of units in each MLP layer for the classifier head.
            trained_segmentator_mdl (tf.keras.Model): Pre-trained model for face parts segmentation.
            seg_conv_params (list of tuples): Specifications for each convolutional layer in the segmentation encoder
                                              in the form (filters, kernel_size, strides).
            enc_output_layer_activ (str): Activation function for the output layer of the RGB encoder. Defaults to 'relu'.
            encoding_dim (int or None): Dimensionality for the dense layer encoding. If None, uses the encoder's default.
            mlp_head_activ (str): Activation function for MLP layers except for the output layer. Defaults to 'relu'.
            mlp_output_layer_activ (str): Activation function for the MLP's output layer. Defaults to 'sigmoid'.
            seg_encoding_dim (int or None): Encoding dimension for the segmentation encoder output. If None, defaults to the output dimension of the conv layers.
            optimizer (str): Optimizer name for model compilation. Defaults to 'adam'.
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
            loss (str): Loss function name for model compilation. Uses either 'mse' or 'binary_crossentropy' for the time being.
        """
        self.input_shape = input_shape
        
        #RGB image encoder parameters
        self.enc_output_layer_activ = enc_output_layer_activ
        self.encoding_dim = encoding_dim

        #Model training parameters
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.loss = loss

        #Trained segmentator Keras model (must be loaded outside this class for now)
        self.trained_segmentator_mdl = trained_segmentator_mdl
        #Setting segmentator model layers to non-trainable
        for layer in self.trained_segmentator_mdl.layers:
            layer.trainable = False

        self.seg_encoder_input_shape = self.trained_segmentator_mdl.output_shape[1:]

        #Segmented face encoder params
        self.encoding_with_same_size = encoding_with_same_size
        self.seg_conv_params = seg_conv_params
        self.seg_encoding_dim = seg_encoding_dim
        self.seg_padding = seg_padding
        self.seg_enc_activation = seg_enc_activation

        #Classifier head parameters
        self.mlp_units = mlp_units
        self.mlp_head_activ=mlp_head_activ
        self.mlp_output_layer_activ=mlp_output_layer_activ

        #Creating RGB image encoder
        self.encoder = self.create_encoder(self.input_shape, enc_output_layer_activ=self.enc_output_layer_activ, encoding_dim=self.encoding_dim)
        self.rgb_encoder_output_shape = self.encoder.output_shape[1]

        # Creating segmented image encoder
        if self.encoding_with_same_size is True:
            if self.seg_encoding_dim is not None:
                #Warn the user, the specified encoding dim will be overwritten by the RGB encoder
                print("###############\nWARNING! Segmentator encoding dim has been specified along with the option to set the segmentation output dim to be equal to the rgb encoder output dim.\nThe segmentation encoding dim will be ignored.\n###############")
            #Setting the segmented encoder output to the same size of the rgb encoder output
            self.encoder_seg = self.create_encoder_seg(self.seg_encoder_input_shape, self.seg_conv_params, self.seg_enc_activation, self.seg_padding, self.rgb_encoder_output_shape)
        else:
            self.encoder_seg = self.create_encoder_seg(self.seg_encoder_input_shape, self.seg_conv_params, self.seg_enc_activation, self.seg_padding, self.seg_encoding_dim)
        self.encoder_seg_output_shape = self.encoder_seg.output_shape[1]
        
        # Getting the input shape for the classifier head and setting it as an attribute of the class for later usage
        self.mlp_head_input_shape = self.rgb_encoder_output_shape + self.encoder_seg_output_shape

        self.concatenated_shape = (self.mlp_head_input_shape,)

        # Creating fake/real classifier mlp head
        self.mlp_head = self.create_mlp_head(self.concatenated_shape, self.mlp_units, mlp_activ=self.mlp_head_activ, mlp_output_layer_activ=self.mlp_output_layer_activ)
        # Creating classifier
        self.classifier = self.create_classifier()
        
    def create_encoder_seg(self, input_shape, conv_params, enc_activation="relu", enc_padding="same", encoding_dim=None, name="encoder_seg"):
        """
        Creates the encoder for the segmentation part using convolutional layers.

        Args:
            input_shape (tuple): The shape of the input data.
            conv_params (list of tuples): Convolutional layer parameters as (filters, kernel_size, strides).
            enc_activation (str): Activation function for convolutional layers. Defaults to 'relu'.
            enc_padding (str): Padding for convolutional layers. Defaults to 'same'.
            encoding_dim (int or None): Final encoding dimension after the convolutional layers. If None, output is flattened.
            name (str): Base name for the layers within this encoder to ensure uniqueness.

        Returns:
            tf.keras.Model: Encoder model for segmentation input.
        """
        inputs = Input(shape=input_shape, name=f"{name}_input")
        x = inputs
        for i, (filters, kernel_size, strides) in enumerate(conv_params):
            conv_layer_name = f"{name}_conv_{i}"
            x = Conv2D(filters, kernel_size, strides=strides, activation=enc_activation, padding=enc_padding, name=conv_layer_name)(x)

        # Flattening to obtain vector representation
        flatten_layer_name = f"{name}_flatten"
        outputs = Flatten(name=flatten_layer_name)(x)

        # Adding a Dense layer if encoding_dim is not None
        if encoding_dim is not None:
            dense_layer_name = f"{name}_dense"
            outputs = Dense(encoding_dim, activation=enc_activation, name=dense_layer_name)(outputs)

        encoder_seg_model = Model(inputs=inputs, outputs=outputs, name=name)
        return encoder_seg_model
    
    def create_encoder(self, input_shape, enc_output_layer_activ="relu", encoding_dim=None, name="encoder"):
        """
        Creates the encoder part using MobileNetV3 Large as a base model.

        Args:
            input_shape (tuple): Shape of the input data.
            enc_output_layer_activ (str): Activation for the output dense layer. Defaults to 'relu'.
            encoding_dim (int or None): Dimensionality of the output encoding. If None, global average pooling output is used.
            name (str): Base name for the layers within this encoder to ensure uniqueness.

        Returns:
            tf.keras.Model: The MobileNetV3Large-based encoder model.
        """
        # Loading MobileNetV3 Large pretrained on ImageNet
        base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_shape, include_preprocessing=True)

        # Setting MobileNetV3 Large layers to non-trainable
        for layer in base_model.layers:
            layer.trainable = False

        # Applying global average pooling
        pooling_layer_name = f"{name}_global_average_pooling"
        encoder_output = GlobalAveragePooling2D(name=pooling_layer_name)(base_model.output)

        # Determine the number of units for the dense layer dynamically if encoding_dim is not provided
        dense_layer_units = encoding_dim if encoding_dim is not None else encoder_output.shape[-1]
        dense_layer_name = f"{name}_dense"

        # Adding a trainable dense layer with specified activation function
        dense_layer = Dense(units=dense_layer_units, activation=enc_output_layer_activ, name=dense_layer_name)(encoder_output)

        # Create and return the encoder model
        encoder_model = Model(inputs=base_model.input, outputs=dense_layer, name=name)
        return encoder_model

    def create_mlp_head(self, concatenated_shape, mlp_units, mlp_activ="relu", mlp_output_layer_activ="sigmoid", name="mlp_head"):
        """
        Creates the MLP classifier head.

        Args:
            mlp_units (list): Number of units in each MLP layer.
            mlp_activ (str): Activation function for MLP layers. Defaults to 'relu'.
            mlp_output_layer_activ (str): Activation for the output layer. Defaults to 'sigmoid'.

        Returns:
            tf.keras.Model: The MLP classifier head model.
        """
        inputs = Input(shape=concatenated_shape, name=f"{name}_input")
        x = inputs
        for i, units in enumerate(mlp_units):
            x = Dense(units, activation=mlp_activ, name=f"{name}_dense_{i}")(x)
        # Using specified activation for output layer
        outputs = Dense(1, activation=mlp_output_layer_activ, name=f"{name}_output")(x)
        mlp_head = Model(inputs=inputs, outputs=outputs, name=name)
        return mlp_head
    
    def create_optimizer(self):
        """
        Creates an instance of the optimizer based on the specified name and learning rate.

        Returns:
            tf.keras.optimizers.Optimizer: The optimizer instance.
        """
        if self.optimizer_name.lower() == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer. Supported options: 'adam', 'sgd'.")
    
    def get_loss(self):
        if self.loss == 'mse':
            return tf.keras.losses.MeanSquaredError()
        elif self.loss == 'binary_crossentropy':
            return tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError("Unsupported loss function: {}".format(self.loss))
    
    def create_classifier(self):
        """
        Constructs the complete classifier model by integrating the RGB and segmentation encoders with the MLP classifier head.

        Returns:
            A compiled tf.keras.Model of the complete GNG classifier.
        """
        input_layer = Input(shape=self.input_shape)
        segmentation_output = self.trained_segmentator_mdl(input_layer)
        seg_encoder_output = self.encoder_seg(segmentation_output)
        rgb_encoder_output = self.encoder(input_layer)
        concatenated_outputs = Concatenate()([seg_encoder_output, rgb_encoder_output])
        classifier_output = self.mlp_head(concatenated_outputs)
        classifier_model = tf.keras.Model(inputs=input_layer, outputs=classifier_output)
        
        # Compile the model with specified optimizer, loss, and metrics
        classifier_model.compile(optimizer=self.create_optimizer(), loss=self.get_loss(), metrics=['accuracy'])
        
        return classifier_model
    
    def call_model(self, input_data):
        """
        Calls the total model on input data.

        Args:
            input_data: Input data to be passed through the model.

        Returns:
            numpy.ndarray: Output predictions.
        """
        return self.classifier(input_data)

    def train_model(self, train_data, train_labels, epochs=10, batch_size=32, class_weights=None):
        """
        Trains the total model.

        Args:
            train_data: Input training data. This must be a numpy array.
            train_labels: Labels for the training data.
            epochs (int): Number of epochs for training. Defaults to 10.
            batch_size (int): Batch size for training. Defaults to 32.
            class_weights (dict): Optional class weights to be passed to the model.
        """
        self.classifier.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,class_weight=class_weights)

    def train_model_gen(self, train_generator, epochs=10, steps_per_epoch=None, class_weights=None,
                        validation_generator=None, validation_steps=None, train_batch_size=32, validation_batch_size=32):
        """
        Trains the total model using an ImageDataGenerator for loading data.

        Args:
            train_generator: An instance of Keras ImageDataGenerator.flow_from_directory for training data.
            epochs (int): Number of epochs for training. Defaults to 10.
            steps_per_epoch (int): Number of steps per epoch. If not specified, it will be calculated based on the length of the generator.
            class_weights (dict): Optional class weights to be passed to the model.
            validation_generator: An instance of Keras ImageDataGenerator.flow_from_directory for validation data. Defaults to None.
            validation_steps (int): Number of steps for validation. If not specified, it will be calculated based on the length of the validation generator.
            train_batch_size (int): Batch size for training. Defaults to 32.
            validation_batch_size (int): Batch size for validation. Defaults to 32.
        """
        if steps_per_epoch is None:
            steps_per_epoch = len(train_generator)
        if validation_generator is not None and validation_steps is None:
            validation_steps = len(validation_generator)

        train_generator.batch_size = train_batch_size
        if validation_generator is not None:
            validation_generator.batch_size = validation_batch_size

        self.classifier.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                            class_weight=class_weights, validation_data=validation_generator,
                            validation_steps=validation_steps)

    def get_encodings(self, input_data):
        """
        Get encodings of the inputs using the encoder part of the model.

        Args:
            input_data: Input data for which encodings are to be obtained.

        Returns:
            numpy.ndarray: Encoded representations of the input data.
        """
        return self.encoder.predict(input_data)
    
    def get_seg_encodings(self, input_data):
        """
        Processes input data through the trained segmentation model and then encodes the segmented data
        using the segmentation encoder.

        The method first applies the trained segmentator model to the input data to generate segmented data.
        Then, it feeds this segmented data into the segmentation encoder to produce encoded representations.

        Args:
            input_data: The input data to be processed. This could be a batch of images or any data compatible
                        with the trained segmentation model. The exact shape and format depend on the model's
                        requirements.

        Returns:
            The encoded representations of the segmented data, produced by the segmentation encoder. The shape
            and type of these encodings depend on the structure of the segmentation encoder model.
        """
        segmented_data = self.trained_segmentator_mdl(input_data)
        return self.encoder_seg(segmented_data)
    
    def save_model(self, file_path):
        """
        Save the model to a .h5 file.

        Args:
            file_path (str): The file path where the model will be saved.
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.classifier.save(file_path)
        print("Model saved to", file_path)

    def load_model(self, file_path):
        """
        Load a saved model from a .h5 file and assign it to the classifier attribute.

        Args:
            file_path (str): The file path from where the model will be loaded.
        """
        if os.path.exists(file_path):
            self.classifier = tf.keras.models.load_model(file_path)
            print("Model loaded from", file_path)
        else:
            print("Error: File does not exist at", file_path)

    def test_class(self):
        print("Testing the model instance")
        print("Model gng_basic is defined with the following parameters:")
        print("Input shape: {} | Encoding output layer activation: {} | Encoding dim: {} | \
               MLP classifier units per hidden layer: {} | MLP classifier activation : {} | MLP output activation :{} | \
               Optimizer: {} | Learning rate: {}".format(self.input_shape,
                                                         self.enc_output_layer_activ,
                                                         self.encoding_dim,
                                                         self.mlp_units,
                                                         self.mlp_head_activ,
                                                         self.mlp_output_layer_activ,
                                                         self.optimizer_name,
                                                         self.learning_rate))
        print("Generating random data for debugging")
        num_samples = 10 
        x_test = np.random.randint(0, 256, size=(num_samples, *self.input_shape), dtype=np.uint8)
        y_test = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
        cw = {0: 2.0, 1: 1.0}
        epochs = 5
        batch_size = 10
        class_weights = {0: 1.0, 1: 2.0}
        print(f"Generated Test Data Shapes:\n X: {x_test.shape}\n Y: {y_test.shape}\n Class Weights: {class_weights}")

        # Training the model
        print(f"Training Model for {epochs} epochs with batch size {batch_size}")
        self.train_model(train_data=x_test, train_labels=y_test, class_weights=cw)
        # Testing the model
        print("Model Predictions:")
        predictions = self.call_model(x_test)
        print(predictions)
        # Getting encodings
        print("Encoding with rgb encoder")
        encodings = self.get_encodings(input_data=x_test)
        print("Encodings:", encodings)
        print("Encoding shape {}".format(encodings.shape))
        print("Encoding with Segmentation encoder the segmentation model outputs")
        seg_encodings = self.get_seg_encodings(input_data=x_test)
        print("Segmentation encodings: ", seg_encodings)
        print("Seg encoding shape {}".format(seg_encodings.shape))

        return

""" ### gng_basic model test code
input_shape = (512, 512, 3)
mlp_layers = [20, 15, 10, 5, 2]
conv_layers = [(20, 5, 2), (20, 3, 1)]
#load segmentator model
segmentator_path="/home/elia/kerasDeeplabFaceseg/deeplabv3plus_face_segmentation_augmentation.h5"
trained_segmentator = tf.keras.models.load_model(segmentator_path)
# Creating an instance of gng_basic model
print("Generating gng_basic model")
    
model = gng_seg(input_shape=input_shape, mlp_units=mlp_layers, trained_segmentator_mdl=trained_segmentator, seg_conv_params=conv_layers, seg_encoding_dim=10, encoding_dim=20, encoding_with_same_size=True)
model.test_class() """

