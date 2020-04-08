from tensorflow import keras
from tensorflow.keras import Input, layers, models
from keras import optimizers, initializers, regularizers, metrics

def get_model(model):
    return model_list[model]

def base_model(x_shape):
    x = Input(shape=(x_shape[1], x_shape[2], x_shape[3]), dtype='float32', name='x')
    
    y = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
    y = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.MaxPooling2D((2,2))(y)
    
    y = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.MaxPooling2D((2,2))(y)
    
    y = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y = layers.MaxPooling2D((2,2))(y)
    
    y = layers.Flatten()(y)
    y = layers.Dense(2048, kernel_initializer='he_normal')(y)
    y = layers.Dense(512, kernel_initializer='he_normal')(y)
    y = layers.Dense(64, kernel_initializer='he_normal')(y)
    y = layers.Dense(1, activation='sigmoid')(y)
    
    return models.Model(inputs=x, outputs=y)


def your_model(x_shape):
    x = keras.Input(shape=(x_shape[1],x_shape[2],x_shape[3]))
    
    ################################
    ##### Make your model here #####    
    ################################
    
    return models.Model(inputs=x, outputs=y)


model_list={
'base_model':base_model,
'your_model':your_model,
}
def  main():
    x_shape=(1,128,128,1)
    model_name = str(input("model : "))
    model = get_model(model_name)(x_shape) 
    model.summary() 
if __name__ == '__main__':
    main()
