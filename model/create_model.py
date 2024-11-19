import tensorflow as tf

class createModel :

  def __init__(self) :
    pass

  def make_nn_model(self, input_shape) : 
    model = tf.keras.Sequential([
      tf.keras.Input(shape=(input_shape,)),

      tf.keras.layers.Dense(64,activation='relu'),
      #tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),

      #addition
      tf.keras.layers.Dense(32,activation='relu'),
      tf.keras.layers.Dropout(0.3),

      tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model
            
