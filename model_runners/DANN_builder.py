"""
Model Builder for Domain-Adversarial Neural Network (DANN)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, GlobalAveragePooling1D

# This is the custom Gradient Reversal Layer
# It acts as an identity in the forward pass
# and multiplies the gradient by -lambda in the backward pass
@tf.custom_gradient
def _gradient_reversal(x, lambda_weight=1.0):
    """
    Implements the gradient reversal layer.
    
    Args:
        x (tf.Tensor): The input tensor.
        lambda_weight (float): The factor to scale the reversed gradient.
        
    Returns:
        tf.Tensor: The identity of x.
        function: The custom gradient function.
    """
    def grad(dy):
        # In the backward pass, multiply the gradient by -lambda_weight
        return -lambda_weight * dy, None  # Gradient for x, None for lambda_weight
    
    # In the forward pass, just return the input tensor (identity)
    return tf.identity(x), grad

class GradientReversalLayer(Layer):
    """
    Custom Keras layer to implement gradient reversal.
    
    This layer is used in the domain discriminator branch of the DANN.
    """
    def __init__(self, lambda_weight=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_weight = lambda_weight
        self.supports_masking = False

    def call(self, inputs):
        """
        Applies the gradient reversal function.
        """
        return _gradient_reversal(inputs, self.lambda_weight)

    def get_config(self):
        """
        Returns the layer configuration.
        """
        config = super(GradientReversalLayer, self).get_config()
        config.update({'lambda_weight': self.lambda_weight})
        return config


def build_dann_model(base_feature_extractor: Model, input_shape: tuple, 
                       learning_rate: float = 0.001, lambda_weight: float = 0.5):
    """
    Builds the complete DANN model with two heads.

    Args:
        base_feature_extractor (Model): A Keras model (e.g., from model_builder.py)
                                        to be used as the feature extractor "trunk".
                                        This model should *not* be compiled and 
                                        *not* have the final classification head.
        input_shape (tuple): The input shape for the model (e.g., (256, 4)).
        learning_rate (float): Learning rate for the Adam optimizer.
        lambda_weight (float): Weight for the domain discriminator loss.

    Returns:
        keras.Model: The compiled DANN model with one input and two outputs.
    """
    
    # --- 1. Define Input ---
    inputs = Input(shape=input_shape)
    
    # --- 2. Feature Extractor (Trunk) ---
    # We pass the inputs through the base feature extractor
    features = base_feature_extractor(inputs)
    
    # --- 3. Branch 1: Signal Classifier ---
    # This branch predicts Signal (1) vs. Background (0)
    # It learns to MINIMIZE this loss
    classifier_branch = GlobalAveragePooling1D()(features)
    classifier_branch = Dense(32, activation="relu", name="classifier_dense")(classifier_branch)
    classifier_output = Dense(1, activation="sigmoid", name="classifier_output")(classifier_branch)

    # --- 4. Branch 2: Domain Discriminator ---
    # This branch predicts Sim (1) vs. Data (0)
    # The Gradient Reversal Layer forces the trunk to MAXIMIZE this loss
    grl_layer = GradientReversalLayer(lambda_weight=lambda_weight)(features)
    domain_branch = GlobalAveragePooling1D()(grl_layer)
    domain_branch = Dense(32, activation="relu", name="domain_dense")(domain_branch)
    domain_output = Dense(1, activation="sigmoid", name="domain_output")(domain_branch)

    # --- 5. Create and Compile Model ---
    # The model has one input and TWO outputs
    model = Model(inputs=inputs, 
                  outputs=[classifier_output, domain_output],
                  name="DANN_Model")
    
    # We provide two losses, one for each output
    # The `loss_weights` apply lambda to the domain loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'classifier_output': 'binary_crossentropy',
            'domain_output': 'binary_crossentropy'
        },
        metrics=['accuracy']
    )
    
    return model
