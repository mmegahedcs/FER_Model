'''
Created on May 3, 2018

@author: Mohamed.Megahed
'''
import cv2

import keras
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau,TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import Adamax
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
tf.logging.set_verbosity(tf.logging.ERROR)

class dccn_manager(object):
    
    image_size=(48,48)
    batch_size = 128
    num_epochs = 170
    input_shape = (48, 48, 1)
    verbose = 1
    num_classes = 7
    patience = 50
    base_path = '\\trained_weights\\' #### A path to save the model weights
    l2_regularization=0.01
     
    def __load_fer2013(self):
        ### Load FER-2013 Dataset
        data = pd.read_csv('\\fer2013.csv')
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        #self.overview(100, data.sample(n=100))
        return faces, emotions
     
    def __preprocess_input(self,x):
        x = x.astype('float32')
        x = x / 255.0
        return x
     
    # parameters
    def get_default_graph(self):
        return tf.get_default_graph() 
    
    
    def extract_from_string(self,pixels):
        pixels = pixels.split(' ')
        pixels = np.array([int(i) for i in pixels])
        return np.reshape(pixels, (48, 48))    

    def extract_image(self,pixels):
        pixels = pixels.as_matrix()[0] # The output is a string
        return self.extract_from_string(pixels)

    def overview(self,total_rows, df):
        fig = plt.figure(figsize=(10,10))
        idx = 0
        for i, row in df.iterrows():
            input_img = self.extract_from_string(row.pixels)
            ax = fig.add_subplot(10,10,idx+1)
            ax.imshow(input_img, cmap=plt.cm.get_cmap('gray'))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
            idx += 1
        plt.show()
        
    def plot_confusion_matrix(self,cm, classes,
                          normalize=True,
                          cmap=plt.cm.get_cmap('Blues')):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
        else:
            title='Confusion matrix'
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    def full_multiclass_report(self,model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False):

        if not binary:
            y_true = np.argmax(y_true,axis=1)
        
        y_pred = model.predict_classes(x, batch_size=batch_size)
        
        print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
        
        print("")
        
        print("Classification Report")
        print(classification_report(y_true,y_pred,digits=5))    
        
        cnf_matrix = confusion_matrix(y_true,y_pred)
        print(cnf_matrix)
        self.plot_confusion_matrix(cnf_matrix,classes=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

    def plot_history(self,history):
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
        
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
        
        ## As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
        
        ## Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        ## Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    
    
    def construct_cnn(self):
        faces, emotions = self.__load_fer2013()
        faces = self.__preprocess_input(faces)
        xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)

       
        train_datagen = ImageDataGenerator(
		rotation_range=2,
        zoom_range=0.1,
        horizontal_flip=True
		)
 
	
        test_datagen = ImageDataGenerator()

        
        model = Sequential()
        model.add(Conv2D(64, (3,3), input_shape=(48,48,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2))) 
        model.add(BatchNormalization())    
        model.add(Conv2D(128, (3,3),kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3),kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.30))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(optimizer   = Adam(lr =  0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.summary()
         
        
        log_file_path = self.base_path + 'emotion1_training.log'
        csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_acc', patience=self.patience)
        reduce_lr =  ReduceLROnPlateau(monitor = "val_acc", factor = 0.1,mode='max', patience = 20, verbose = 1,cooldown = 0)
        trained_models_path = self.base_path + 'cnn1_model_weights'
        model_names = trained_models_path + '.{epoch:04d}--{val_loss:.4f}--{val_acc:.4f}.h5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_acc', verbose=1,save_best_only=True)

        callbacks = [model_checkpoint, csv_logger, early_stop,reduce_lr]
         
        history= model.fit_generator(train_datagen.flow(xtrain, ytrain,self.batch_size),
                                steps_per_epoch=len(xtrain) / self.batch_size,
                                epochs=self.num_epochs, verbose=1, callbacks=callbacks,
                                validation_data=(xtest,ytest))
        
        self.full_multiclass_report(model,
                       xtest,
                       ytest,
                       classes=self.num_classes)
       
        self.plot_history(history)

        
        return model
    
if __name__ == "__main__": 
 
   config = tf.ConfigProto(device_count={'GPU': 0 , 'CPU': 56})
   cnn_man = dccn_manager()
   model = cnn_man.construct_cnn()
