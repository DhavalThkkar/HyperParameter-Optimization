#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:40:03 2017

@author: thakkar_
"""

class HyperParams(object):
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    import numpy as np
    num_classes = 2
    epochs = 12
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    def __init__(self,batch_size = 128, epochs = 12,x_train = X_train, x_test = X_test, y_train = Y_train , y_test = Y_test):
        self.batch_size = batch_size
        self.epochs = epochs
        import keras
        from keras.datasets import mnist
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D
        from keras import backend as K
        import numpy as np
        num_classes = 2
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        #Only look at 3s and 8s
        train_picks = np.logical_or(y_train==2,y_train==7)
        test_picks = np.logical_or(y_test==2,y_test==7)
        
        x_train = x_train[train_picks]
        x_test = x_test[test_picks]
        y_train = np.array(y_train[train_picks]==7,dtype=int)
        y_test = np.array(y_test[test_picks]==7,dtype=int)
        
        
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
            
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')
                
            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            model = Sequential()
            model.add(Conv2D(4, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
            model.add(Conv2D(8, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='softmax'))
            
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
            
            model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      verbose=1,
                      validation_data=(x_test, y_test))
            score = model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
    
    def batchOpt(self,x_train,y_train,x_test,y_test):
        history = []
        import keras
        from keras.datasets import mnist
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D
        from keras import backend as K
        import numpy as np
        for x in self.batch_size:
            model = Sequential()
            model.add(Conv2D(4, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
            model.add(Conv2D(8, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='softmax'))
            
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
            
            batch_history =  model.fit(x_train, y_train,
                             batch_size=x,
                             epochs=epochs,
                             verbose=1,
                             validation_data=(x_test, y_test))
            score = model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            history.append(batch_history)
       
batch = [128, 256, 512]
hp = HyperParams(batch_size=batch)
hp.batchOpt()