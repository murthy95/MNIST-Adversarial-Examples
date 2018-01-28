import numpy as  np
import os
from mnist import MNIST
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Activation
from keras.layers.core import Reshape, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.metrics import categorical_accuracy
from keras.layers.normalization import BatchNormalization
from keras import regularizers


def load_data():
    mndata = MNIST('./data')
    train_images, train_labels = mndata.load_training()
    train_images = np.array(train_images)
    train_labels =np.array(train_labels)
    test_images, test_labels = mndata.load_testing()
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels

def numpy_onehot(arr):
    one_hot = np.zeros((arr.shape[0],10))
    for i in range(arr.shape[0]):
        one_hot[i, arr[i]] = 1
    return one_hot



class cnn_model(object):

    def __init__(self, input_shape=(28,28,1), n_classes=10 , l2_reg=0.005):
        self.input_shape = input_shape
        self.output_shape = n_classes
        self.l2_reg = l2_reg

    def return_model_1(self):

        '''
        In problem 1 the model has only one input layer a conv layer and an output softmax layer
        '''
        inputs = Input(self.input_shape)
        conv1 = Conv2D(32, (3,3), activation = 'linear', strides = (1,1), padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(self.l2_reg))(inputs)
        # padding same indicats that the output sahpe is same as input in this case is same as using zero padding of 1
        print "conv1 shape:",conv1.shape
        conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
        print "pool1 shape:",pool1.shape
        flatten = Flatten()(pool1)
        print "flatten shape:",flatten.shape
        dense1 = Dense(10, activation='softmax' , kernel_regularizer=regularizers.l2(self.l2_reg))(flatten)
        print "output shape:",dense1.shape

        model = Model(input = inputs, output = dense1)
        model.compile(optimizer = Adam(lr = 1e-4, decay = 0.0005), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model

    def return_model_2(self):

        '''
        In problem 1 the model has only one input layer a conv layer and an output softmax layer
        '''
        inputs = Input(self.input_shape)
        conv1 = Conv2D(32, (3,3), activation = 'linear', strides = (1,1), padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(self.l2_reg))(inputs)
        # padding same indicats that the output sahpe is same as input in this case is same as using zero padding of 1
        print "conv1 shape:",conv1.shape
        conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
        print "pool1 shape:",pool1.shape

        conv2 = Conv2D(32, (3,3), activation = 'linear', strides = (1,1), padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(self.l2_reg))(pool1)
        # padding same indicats that the output sahpe is same as input in this case is same as using zero padding of 1
        print "conv2 shape:",conv2.shape
        conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        print "pool1 shape:",pool2.shape

        flatten = Flatten()(pool2)
        print "flatten shape:",flatten.shape
        dense1 = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(self.l2_reg))(flatten)
        print "output shape:",dense1.shape

        model = Model(input = inputs, output = dense1)
        model.compile(optimizer = Adam(lr = 1e-4, decay = 0.0005), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model

    def return_model_3(self):

        '''
        In problem 1 the model has only one input layer a conv layer and an output softmax layer
        '''
        inputs = Input(self.input_shape)
        conv1 = Conv2D(32, (3,3), activation = 'linear', strides = (1,1), padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(self.l2_reg))(inputs)
        # padding same indicats that the output sahpe is same as input in this case is same as using zero padding of 1
        print "conv1 shape:",conv1.shape
        conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
        print "pool1 shape:",pool1.shape

        conv2 = Conv2D(32, (3,3), activation = 'linear', strides = (1,1), padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(self.l2_reg))(pool1)
        # padding same indicats that the output sahpe is same as input in this case is same as using zero padding of 1
        print "conv2 shape:",conv2.shape
        conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        print "pool1 shape:",pool2.shape

        flatten = Flatten()(pool2)
        print "flatten shape:",flatten.shape
        dense1 = Dense(500, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(flatten)
        dense2 = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(self.l2_reg))(dense1)
        print "output shape:",dense2.shape

        model = Model(input = inputs, output = dense2)
        model.compile(optimizer = Adam(lr = 1e-4, decay = 0.0005), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model

    def train_cnn(self):

        x_train, y_train, x_test, y_test = load_data()
        x_train = np.reshape(np.array(x_train), (x_train.shape[0],28,28,1))
        x_test = np.reshape(np.array(x_test), (x_test.shape[0],28,28,1))
        y_train = numpy_onehot(np.reshape(np.array(y_train), (y_train.shape[0],1)))
        y_test = numpy_onehot(np.reshape(np.array(y_test), (y_test.shape[0],1)))

        model = self.return_model_3()

        if os.path.exists('q3.hdf5'):
            model.load_weights('q3.hdf5')

        model_checkpoint = ModelCheckpoint('q3.hdf5', monitor='loss',verbose=1, save_best_only=True)
        history = model.fit(x_train, y_train, batch_size=32, validation_split= 0.33, epochs=30, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        test_score = model.evaluate(x_test, y_test, verbose=1)

        print "final test accuracy is:",test_score

if __name__ =='__main__':
    cnn = cnn_model()
    cnn.train_cnn()
