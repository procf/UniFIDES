import tensorflow as tf

class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model """
    def __init__(self, output_dim=1, num_hidden_layers=3, num_neurons_per_layer=16, activation='tanh', kernel_initializer='glorot_normal', **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                       for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        
    def call(self, X):
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)