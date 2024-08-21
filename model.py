import tensorflow as tf
from tensorflow import keras
class N_Network():
    def __init__(self, hidden_layers = [256], hidden_activation= tf.nn.relu):
        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
    


    def create_model(self,no_inputs, no_outputs):
        input = tf.keras.Input(no_inputs)
        layer = tf.keras.layers.Dense(self.hidden_layers[0],kernel_initializer= tf.keras.initializers.RandomNormal(stddev=0.01), activation=self.hidden_activation)(input)
        for no_neurons in self.hidden_layers[1:]:
            layer = tf.keras.layers.Dense(no_neurons,kernel_initializer= tf.keras.initializers.RandomNormal(stddev=0.01), activation=self.hidden_activation)(layer)   
        output = tf.keras.layers.Dense(no_outputs, activation=tf.nn.relu)(layer)
        return tf.keras.Model(inputs=input, outputs=output)

class Dueling_Network():
    def __init__(self, hidden_layers = [256], hidden_activation= tf.nn.relu):
        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
    

    def create_model(self,no_inputs, no_outputs):
        input = tf.keras.Input(no_inputs)
        layer = tf.keras.layers.Dense(self.hidden_layers[0],kernel_initializer= tf.keras.initializers.RandomNormal(stddev=0.01), activation=self.hidden_activation)(input)
        for no_neurons in self.hidden_layers[1:]:
            layer = tf.keras.layers.Dense(no_neurons,kernel_initializer= tf.keras.initializers.RandomNormal(stddev=0.01), activation=self.hidden_activation)(layer)   
        value = tf.keras.layers.Dense(no_outputs, activation=tf.nn.relu)(layer)
        advantage = tf.keras.layers.Dense(no_outputs, activation=tf.nn.relu)(layer)
        output = value + (advantage -tf.math.reduce_mean(advantage, axis=1, keepdims=True))
        return tf.keras.Model(inputs=input, outputs=output)

      
