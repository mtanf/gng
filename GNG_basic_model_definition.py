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
from tensorflow.keras.layers import Dense, Input,GlobalAveragePooling2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

#Creating basic GNG classifier (MobilenetV3Large encoder + MLP classifier on top applied to RGB images for fake vs réal classification)
class gng_basic:
    """
    A class for building the basic version of GNG: a MobileNetv3Large and an MLP classifier.

    Attributes:
        encoder: The encoder part of the model (MobileNetV3Large, non-trainable).
        mlp_head: The classifier head of the model (Simple MLP with specified activation functions).
        classifier: The total classifier model combining encoder and classifier head.
    """

    def __init__(self, input_shape, mlp_units, callbacks_list,
                 enc_output_layer_activ="tanh", encoding_dim=None, use_pooling=False,
                   mlp_head_activ="tanh", mlp_output_layer_activ="sigmoid",
                     optimizer='adam', learning_rate=0.001, loss='mse'):
        """
        Initializes the gng_basic model.

        Args:
            input_shape (tuple): The shape of the input data.
            mlp_units (list): A list of integers specifying the number of units in each layer of the MLP classifier.
            enc_output_layer_activ (str): The activation function for the output layer of the encoder. Defaults to "tanh".
            encoding_dim (int, optional): The dimensionality of the encoding. If None, the dimensionality is determined dynamically based on the shape of the output tensor. Defaults to None.
            mlp_head_activ (str): The activation function for hidden layers of the MLP classifier. Defaults to "tanh".
            mlp_output_layer_activ (str): The activation function for the output layer of the classifier. Defaults to "sigmoid".
            optimizer (str): Name of the optimizer to use for training. Defaults to 'adam'.
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
            loss (str): Name of the loss function to use for training. Defaults to 'mse'.
        """
        self.input_shape = input_shape
        self.enc_output_layer_activ = enc_output_layer_activ
        self.encoding_dim = encoding_dim
        self.mlp_units = mlp_units
        self.mlp_head_activ=mlp_head_activ
        self.mlp_output_layer_activ=mlp_output_layer_activ
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.use_pooling = use_pooling
        self.callbacks_list = callbacks_list

        self.encoder = self.create_encoder(self.input_shape, enc_output_layer_activ=self.enc_output_layer_activ, encoding_dim=self.encoding_dim)
        self.mlp_head = self.create_mlp_head(self.mlp_units, mlp_activ=self.mlp_head_activ, mlp_output_layer_activ=self.mlp_output_layer_activ)
        self.classifier = self.create_classifier()
        print(self.classifier.summary())

    def create_encoder(self, input_shape, enc_output_layer_activ="tanh", encoding_dim=None):
        """
        Creates the encoder part of the model using MobileNetV3 Large.

        Args:
            input_shape (tuple): The shape of the input data.
            enc_output_layer_activ (str): The activation function for the output layer of the encoder. Defaults to "tanh".
            encoding_dim (int, optional): The dimensionality of the encoding. If None, the dimensionality is determined dynamically based on the shape of the output tensor. Defaults to None.

        Returns:
            tf.keras.Model: The encoder model.
        """
        #Defining input layer
        input = Input(shape = input_shape)
        # Loading MobileNetV3 Large pretrained on ImageNet
        base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape = input_shape, include_preprocessing=True)

        '''
        # Setting MobileNetV3 Large layers to non-trainable
        for layer in base_model.layers:
            layer.trainable = False
        '''

        x = base_model(input)

        # Applying global average pooling or flattening
        
        if self.use_pooling:
            encoder_output = GlobalAveragePooling2D()(x)
        else:
            encoder_output = Flatten()(x)

        
        if encoding_dim is None:
            # Determine the number of units for the dense layer dynamically
            dense_layer_units = encoder_output.shape[-1]
        else:
            dense_layer_units = encoding_dim
        
        # Adding a trainable dense layer with specified activation
        encoder_output = Dense(units=dense_layer_units, activation=enc_output_layer_activ)(encoder_output)
        
        # Create the encoder model
        encoder_model = Model(inputs=input, outputs=encoder_output)

        return encoder_model

    def create_mlp_head(self, mlp_units, mlp_activ="tanh", mlp_output_layer_activ="sigmoid"):
        """
        Creates the classifier head part of the model.

        Args:
            mlp_units (list): A list of integers specifying the number of units in each layer of the MLP classifier.
            mlp_activ (str): The activation function for hidden layers of the MLP classifier. Defaults to "tanh".
            mlp_output_layer_activ (str): The activation function for the output layer of the classifier. Defaults to "sigmoid".

        Returns:
            tf.keras.Model: The classifier head model.
        """
        inputs = Input(shape=self.encoder.output_shape)
        x = inputs
        for units in mlp_units:
            x = Dense(units, activation=mlp_activ)(x)
        # Using specified activation for output layer
        outputs = Dense(1, activation=mlp_output_layer_activ)(x)
        mlp_head = Model(inputs=inputs, outputs=outputs)
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
        Creates the total classifier model by connecting the encoder and classifier head.
        
        Returns:
            tf.keras.Model: The total classifier model.
        """
        classifier_input = self.encoder.input
        classifier_output = self.mlp_head(self.encoder.output)
        classifier_model = tf.keras.Model(inputs=classifier_input, outputs=classifier_output)
        
        # Compile the model with specified optimizer, loss, and metrics
        classifier_model.compile(optimizer=self.create_optimizer(), loss=self.get_loss(), metrics=['accuracy',
                                                                                                   tf.keras.metrics.Precision(name='precision'),
                                                                                                   tf.keras.metrics.Recall(name='recall'),
                                                                                                   tf.keras.metrics.F1Score(average="weighted",name='F1')])
        
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
        self.training_history = self.classifier.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,class_weight=class_weights, callbacks=self.callbacks_list)

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
                            validation_steps=validation_steps, callbacks=self.callbacks_list)

    def get_encodings(self, input_data):
        """
        Get encodings of the inputs using the encoder part of the model.

        Args:
            input_data: Input data for which encodings are to be obtained.

        Returns:
            numpy.ndarray: Encoded representations of the input data.
        """
        return self.encoder.predict(input_data)

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
        num_samples = 10  # Number of samples
        x_test = np.random.randint(0, 256, size=(num_samples, *self.input_shape), dtype=np.uint8)
        y_test = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
        cw = {0: 2.0, 1: 1.0}
        epochs = 100
        batch_size = 64
        
        print("Test class weights: {}".format(cw))
        print("Shape of generated test X data:", x_test.shape)
        print("Shape of generated test Y data:", y_test.shape)
        # Training the model
        print("Training the model")
        print("Training epochs: {} | Batch size: {}".format(epochs, batch_size))
        self.train_model(train_data=x_test, train_labels=y_test, class_weights=cw)
        # Testing the model
        print("Predicting with trained model")
        predictions = self.call_model(input_data=x_test)
        print("Predictions:", predictions)
        print("Predictions shape: {}".format(predictions.shape))
        # Getting encodings
        print("Encoding with encoder")
        encodings = self.get_encodings(input_data=x_test)
        print("Encodings:", encodings)
        print("Encoding shape {}".format(encodings.shape))

        return
#
# ### gng_basic model test code
# input_shape = (224, 224, 3)
# mlp_layers = [20, 15, 10, 5, 2]
#
#
# # Creating an instance of gng_basic model
# print("Generating gng_basic model")
# model = gng_basic(input_shape=input_shape, mlp_units=mlp_layers)
# model.test_class()

