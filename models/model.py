
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, Dropout

def build_esrcnn(scale_factor=2, num_filters=[64, 32], filter_sizes=[9, 1, 5]):

    model = Sequential()
    model.add(UpSampling2D(size=(scale_factor, scale_factor), input_shape=(None, None, 3)))

    model.add(Conv2D(num_filters[0], (filter_sizes[0], filter_sizes[0]), activation='relu', padding='same'))
    model.add(Dropout(0.04))

    model.add(Conv2D(num_filters[1], (filter_sizes[1], filter_sizes[1]), activation='relu', padding='same'))
    model.add(Dropout(0.04))

    model.add(Conv2D(3, (filter_sizes[2], filter_sizes[2]), padding='same'))
    return model
