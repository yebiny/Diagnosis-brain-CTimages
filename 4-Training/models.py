from tensorflow import keras
from tensorflow.keras import layers, models
def get_model(model):
    return model_list[model]


def your_model(x_shape):
    x = keras.Input(shape=(x_shape[1],x_shape[2],x_shape[3]))
    
    ################################
    ##### Make your model here #####    
    ################################
    
    return models.Model(inputs=x, outputs=y)


def base_model(x_shape):
  
  x = keras.Input(shape=(x_shape[1],x_shape[2],x_shape[3]))
  
  y = layers.Conv2D(8,(3,3), padding='same')(x)
  y = layers.Activation('relu')(y)
  y = layers.MaxPool2D(pool_size=(3,3))(y)
  
  y = layers.Conv2D(16,(3,3), padding='same')(y)
  y = layers.Activation('relu')(y)
  y = layers.MaxPool2D(pool_size=(3,3))(y)
  
  y = layers.Flatten()(y)
  
  y = layers.Dense(1024)(y)
  y = layers.BatchNormalization()(y)
  y = layers.Activation('relu')(y)

  y = layers.Dense(128)(y)
  y = layers.BatchNormalization()(y)
  y = layers.Activation('relu')(y)
  
  y = layers.Dense(32)(y)
  y = layers.BatchNormalization()(y)
  y = layers.Activation('relu')(y)
  
  y = layers.Dense(1)(y)
  y = layers.Activation('sigmoid')(y)
  
  return models.Model(inputs=x, outputs=y)


model_list={
'base_model':base_model,
'your_model':your_model
}


