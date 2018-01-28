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

class Addition2D(Layer):np.array(
    def build(self,input_shape=(None,28,28,1)):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(28,28,1),
                                      initializer='zeros',
                                      trainable=True)
        super(Addition2D, self).build(input_shape)

    def call(self, x):
        return K.clip((x + self.kernel), 0, 255)


    def compute_output_shape(self, input_shape=(None,28,28,1)):
        return (input_shape[0], 28,28,1)

class custom_SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(custom_SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr *K.sign(g)   # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * K.sign(g)
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
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

        for class_label in range(10):
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
            dense2 = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(self.l2_reg),trainable=False)(dense1)
            print "output shape:",dense2.shape

            model2 =Model(input =in1, output=dense2)

            model2.layers[2].set_weights(model1.layers[1].get_weights())
            model2.layers[3].set_weights(model1.layers[2].get_weights())
            model2.layers[6].set_weights(model1.layers[5].get_weights())
            model2.layers[7].set_weights(model1.layers[6].get_weights())
            model2.layers[11].set_weights(model1.layers[10].get_weights())
            model2.layers[12].set_weights(model1.layers[11].get_weights())

            y_train_adv = np.zeros((y_train.shape[0],10))
            y_train_adv[:,class_label] =1

            y_test_adv = np.zeros((y_test.shape[0],10))
            y_test_adv[:,class_label] =1

            model_noise = Model(input=in1, output=inputs)
            custom_optimizer = custom_SGD(lr = 0.03,decay=0.0001)

            noise_ = noisetracker()

            model2.compile(optimizer = custom_optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
            history = model2.fit(x_train, y_train_adv, batch_size=32, validation_split= 0.33,epochs=7, verbose=1, shuffle=True) #, callbacks=[noise_])
            #
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.title("Training and Validation accuracy for class "+str(class_label))
            plt.savefig('q2_adv_err'+str(class_label)+'.png')
            plt.clf()

            test_score = model2.evaluate(x_test, y_test_adv, verbose=1)
            print "test_accuracy",test_score

            adv_noise = np.array(model2.layers[1].get_weights())
            filename = 'class_'+str(class_label)+'.npy'
            nos = adv_noise
            nos = nos.reshape((28,28))
            print nos.shape
            plt.imshow(nos)
            plt.title('Adversarial Noise for Class '+str(class_label))
            plt.savefig('q2_adv_noise'+str(class_label)+'.png')
            plt.clf()
            np.save(filename,adv_noise)
            adv_noise = np.load(filename)
            model2.layers[1].set_weights(adv_noise)
            test_score = model2.evaluate(x_test, y_test_adv, verbose=1)
            print "test_accuracy",test_score
            print "x_test_shape",x_test.shape
            x_test_adv = np.add(x_test,[adv_noise.reshape((28,28,1))])
            x_test_adv.clip(min=0,max=255)
            print "x_test_adv_shape",x_test_adv.shape
            test_score = model1.evaluate(x_test_adv, y_test_adv, verbose=1)
            print "test_accuracy",test_score


        np.random.seed(10)
        index = np.random.randint(1000,size=10)
        fig =plt.figure()

        for i in range(11):

            if i==0:
                for j in range(10):
                    plt.subplot(11,11,j+2)
                    plt.imshow(x_test[index[j]].reshape((28,28)))
                    plt.title('True Label: '+str(np.argmax(y_test[index[j]])),fontsize=5)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
            else:
                filename = 'class_'+str(i-1)+'.npy'
                adv_noise = np.load(filename).reshape((28,28))
                print "noise_shape",adv_noise.shape
                for j in range(11):

                    if(j==0):
                        plt.subplot(11,11,11*(i)+1)
                        plt.imshow(adv_noise)
                        plt.title('Noise for class: '+str(i-1),fontsize=5)
                        plt.xticks([])
                        plt.yticks([])
                        plt.axis('off')

                    else:
                        image_adv = np.add(x_test[index[j-1]].reshape((28,28)), adv_noise)
                        plt.subplot(11,11,11*(i)+j+1)
                        image_adv.clip(min=0,max=255)
                        plt.imshow(image_adv)
                        # print str(np.argmax(model1.predict(np.array([x_test[index[j-1]]]))))+str(np.argmax(model1.predict(np.reshape(image_adv,(1,28,28,1)))))
                        plt.title('predicted label: '+str(np.argmax(model1.predict(image_adv.reshape((1,28,28,1))))),fontsize=5,y=0.8)
                        plt.xticks([])
                        plt.yticks([])
                        plt.axis('off')
        # plt.subplot(3,1,1)
        # test=x_test[index[0]].reshape((28,28))
        # plt.imshow(test)
        # filename = 'class_'+str(0)+'.npy'
        # adv_noise = np.load(filename).reshape((28,28))
        # plt.subplot(3,1,2)
        # plt.imshow(adv_noise)
        # plt.subplot(3,1,3)
        # image_adv = np.add(adv_noise, test)
        # plt.imshow(np.add(adv_noise, test))
        # plt.title('predicted label: '+str(np.argmax(model1.predict(image_adv.reshape((1,28,28,1))))),fontsize=5)
        plt.axis('off')
        plt.show()

if __name__ =='__main__':
    cnn = cnn_model()
    cnn.train_cnn()
