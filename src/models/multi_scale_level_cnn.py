from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Dropout, Activation, LSTM, GRU, \
    TimeDistributed, Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D, \
    BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, \
    ZeroPadding2D, Reshape, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D

from tensorflow.keras.models import Model


def base_conv_block(num_conv_filters, kernel_size, initializer):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        out = Convolution2D(num_conv_filters, kernel_size, padding='same', kernel_initializer=initializer)(x)
        return out

    return f


def multi_scale_block(num_conv_filters, initializer):
    def f(input_):
        branch1x1 = base_conv_block(num_conv_filters, 1, initializer)(input_)

        branch3x3 = base_conv_block(num_conv_filters, 1, initializer)(input_)
        branch3x3 = base_conv_block(num_conv_filters, 3, initializer)(branch3x3)

        branch5x5 = base_conv_block(num_conv_filters, 1, initializer)(input_)
        branch5x5 = base_conv_block(num_conv_filters, 5, initializer)(branch5x5)

        branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_)
        branchpool = base_conv_block(num_conv_filters, 1, initializer)(branchpool)

        out = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=-1)
        #         out = base_conv_block(num_conv_filters, 1)(out)
        return out

    return f


def dense_block(num_dense_blocks, num_conv_filters, initializer):
    def f(input_):
        x = input_
        for _ in range(num_dense_blocks):
            out = multi_scale_block(num_conv_filters, initializer)(x)
            x = concatenate([x, out], axis=-1)
        return x

    return f


def transition_block(num_conv_filters, initializer):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        x = Convolution2D(num_conv_filters, 1, kernel_initializer=initializer)(x)
        out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return out

    return f


def multi_scale_level_cnn(input_shape, num_dense_blocks, num_conv_filters, num_classes, initializer):
    model_input = Input(shape=input_shape)

    x = Convolution2D(num_conv_filters, 3, padding='same', kernel_initializer=initializer)(model_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    x = dense_block(num_dense_blocks, num_conv_filters, initializer)(x)
    x = transition_block(num_conv_filters, initializer)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    model_output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=model_input, outputs=model_output)

    return model
