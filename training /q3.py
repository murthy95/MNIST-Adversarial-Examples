'''
visualizing convolutional neural network
using gradient backpass
'''

import numpy as  np
import os
from mnist import MNIST
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Activation,Layer
from keras.layers.core import Reshape, Flatten, Dense
from keras.optimizers import SGD, Adam, Optimizer
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.metrics import categorical_accuracy
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
import keras

class Addition2D(Layer):
    def build(self,input_shape=(None,28,28,1)):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(28,28,1),
                                      initializer= keras.initializers.RandomNormal(mean=128, stddev=10, seed=None),
                                      trainable=True)
        super(Addition2D, self).build(input_shape)

    def call(self, x):
        return K.clip((x + self.kernel), 0, 255)


    def compute_output_shape(self, input_shape=(None,28,28,1)):
        return (input_shape[0], 28,28,1)

class custom_SGD(Optimizer):

    def __init__(self, lr=0.01, alpha= 0.0001, **kwargs):
        super(custom_SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.alpha = K.variable(alpha, name='alpha')


    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        # if self.initial_decay > 0:
            # lr *= (1. / (1. + self.decay * K.cast(self.iterations,
            #                                       K.dtype(self.decay))))

        for p, g in zip(params, grads):
            # lr = 1/K.std(g)
            g /= (K.sqrt(K.mean(K.square(g))) + 1e-5)
            new_p = (1- self.alpha)*p + lr*g

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'alpha': float(K.get_value(self.alpha))}
        base_config = super(custom_SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

def cost_function(y_true, y_pred):
    return y_pred[0,3,3,2]

def cost_function_a(y_true, y_pred):
    return y_pred[0,9]

class noisetracker(keras.callbacks.Callback):
    def on_epoch_begin(self, batch, logs={}):
        print np.array(self.model.layers[1].get_weights())

class cnn_model:

    def __init__(self, input_shape=(28,28,1), n_classes=10, l2_reg=0.005):
        self.input_shape = input_shape
        self.output_shape = n_classes
        self.l2_reg = l2_reg

    def return_model_3(self):

        '''
        In problem 1 the model has only one input layer a conv layer and an output softmax layer
        '''
        inputs = Input(self.input_shape)
        print "input_shape",inputs.shape
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
        model.compile(optimizer = Adam(lr = 0.0001, decay = 0.0005), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model

    def train_cnn(self):

        x_train, y_train, x_test, y_test = load_data()
        x_train = np.reshape(np.array(x_train), (x_train.shape[0],28,28,1))
        x_test = np.reshape(np.array(x_test), (x_test.shape[0],28,28,1))
        y_train = numpy_onehot(np.reshape(np.array(y_train), (y_train.shape[0],1)))
        y_test = numpy_onehot(np.reshape(np.array(y_test), (y_test.shape[0],1)))

        model1 = self.return_model_3()

        if os.path.exists('q3.hdf5'):
            model1.load_weights('q3.hdf5')

        for class_label in range(1):
            class_label = 9
            in1 = Input(self.input_shape)
            print "input_shape",in1.shape
            inputs = Addition2D()(in1)
            print "input_shape",inputs.shape
            conv1 = Conv2D(32, (3,3), activation = 'linear', strides = (1,1), padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(self.l2_reg), trainable=False)(inputs)
            # padding same indicats that the output sahpe is same as input in this case is same as using zero padding of 1
            print "conv1 shape:",conv1.shape
            conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, trainable=False)(conv1)
            conv1 = Activation('relu')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
            print "pool1 shape:",pool1.shape

            conv2 = Conv2D(32, (3,3), activation = 'linear', strides = (1,1), padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(self.l2_reg),trainable=False)(pool1)
            # padding same indicats that the output sahpe is same as input in this case is same as using zero padding of 1
            print "conv2 shape:",conv2.shape
            conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, trainable=False)(conv2)
            conv2 = Activation('relu')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
            print "pool1 shape:",pool2.shape

            flatten = Flatten()(pool2)
            print "flatten shape:",flatten.shape
            dense1 = Dense(500, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg),trainable=False)(flatten)
            dense2 = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg),trainable=False)(dense1)
            out = Activation('softmax')(dense2)
            print "output shape:",dense2.shape


            # model2b =Model(input =in1, output=pool2)
            model2 =Model(input =in1, output=dense2)

            model2.layers[2].set_weights(model1.layers[1].get_weights())
            model2.layers[3].set_weights(model1.layers[2].get_weights())
            model2.layers[6].set_weights(model1.layers[5].get_weights())
            model2.layers[7].set_weights(model1.layers[6].get_weights())
            model2.layers[11].set_weights(model1.layers[10].get_weights())
            model2.layers[12].set_weights(model1.layers[11].get_weights())

            x_white = np.zeros((1,28,28,1))
            y_white = np.zeros(10)
            y_white[class_label] = 1
            custom_optimizer = custom_SGD(lr = 0.04)

            model2.compile(optimizer = custom_optimizer, loss = cost_function_a)
            history = model2.fit(x_white, np.array([y_white]),epochs=15000, verbose=1, shuffle=True) #, callbacks=[noise_])

            plt.plot(history.history['loss'])
            plt.title('cost over iterations')
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.savefig('q3_cost'+str(class_label)+'.png')
            plt.clf()
            # plt.show()

            adv_noise = np.array(model2.layers[1].get_weights())
            filename = 'q3_class_'+str(class_label)+'.npy'
            nos = adv_noise
            nos = nos.reshape((28,28))
            print nos.shape
            plt.imshow(nos)
            plt.title("x_init for class "+str(class_label))
            plt.savefig('q3_xinit'+str(class_label)+'.png')
            plt.clf()
            # plt.show()

            # model2.fit(x_white, y_white,epochs=15000, verbose=1, shuffle=True) #, callbacks=[noise_])
            # adv_noise = np.array(model2.layers[1].get_weights())
            # filename = 'q3b_class_'+str(class_label)+'.npy'
            # nos = adv_noise
            # nos = nos.reshape((28,28))
            # print nos.shape
            # plt.imshow(nos)
            # plt.title("feature map for channel "+str(class_label))
            # plt.savefig('q3b_xinit'+str(class_label)+'.png')
            # plt.clf()





if __name__ =='__main__':
    cnn = cnn_model()
    cnn.train_cnn()
