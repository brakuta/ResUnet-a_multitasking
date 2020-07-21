import tensorflow.keras.models as KM
import tensorflow.keras as KE
import tensorflow.keras.layers as KL
#import tensorflow.keras.engine as KE
import tensorflow.keras.backend as KB
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf

from ResUnet_a.config import UnetConfig
import utils
import math
import numpy as np
class Resunet_a(object):
    def __init__(self, input_shape,config=UnetConfig()):
        self.config = config
        print(f"Input shape: {input_shape}")
        self.img_height, self.img_width, self.img_channel = input_shape
        self.model = self.build_model_ResUneta()

    def build_model_ResUneta(self):
        def Tanimoto_loss(label,pred):
            square=tf.square(pred)
            sum_square=tf.reduce_sum(square,axis=-1)
            product=tf.multiply(pred,label)
            sum_product=tf.reduce_sum(product,axis=-1)
            denomintor=tf.subtract(tf.add(sum_square,1),sum_product)
            loss=tf.divide(sum_product,denomintor)
            loss=tf.reduce_mean(loss)
            return 1.0-loss

        def Tanimoto_dual_loss(label,pred):
            loss1=Tanimoto_loss(pred,label)
            pred=tf.subtract(1.0,pred)
            label=tf.subtract(1.0,label)
            loss2=Tanimoto_loss(label,pred)
            loss=(loss1+loss2)/2

        def ResBlock(input,filter,kernel_size,dilation_rates,stride):
            def branch(dilation_rate):
                x=KL.BatchNormalization()(input)
                x=KL.Activation('relu')(x)
                x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate,padding='same')(x)
                x=KL.BatchNormalization()(x)
                x=KL.Activation('relu')(x)
                x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate,padding='same')(x)
                return x
            out=[]
            for d in dilation_rates:
                out.append(branch(d))
            if len(dilation_rates)>1:
                out=KL.Add()(out)
            else:
                out=out[0]
            return out
        def PSPPooling(input,filter):
            print('[DEBUG]'*10)
            print(input.shape)
            print('[DEBUG]'*10)
            x1=KL.MaxPooling2D(pool_size=(2,2), padding='same')(input)
            x2=KL.MaxPooling2D(pool_size=(4,4), padding='same')(input)
            x3=KL.MaxPooling2D(pool_size=(8,8), padding='same')(input)
            x4=KL.MaxPooling2D(pool_size=(16,16), padding='same')(input)
            x1=KL.Conv2D(int(filter/4),(1,1), padding='same')(x1)
            x2=KL.Conv2D(int(filter/4),(1,1), padding='same')(x2)
            x3=KL.Conv2D(int(filter/4),(1,1), padding='same')(x3)
            x4=KL.Conv2D(int(filter/4),(1,1), padding='same')(x4)
            print(x1.shape)
            print(x2.shape)
            print(x3.shape)
            print(x4.shape)
            x1=KL.UpSampling2D(size=(2,2))(x1)
            x2=KL.UpSampling2D(size=(4,4))(x2)
            x3=KL.UpSampling2D(size=(8,8))(x3)
            x4=KL.UpSampling2D(size=(16,16))(x4)
            print(x1.shape)
            print(x2.shape)
            print(x3.shape)
            print(x4.shape)
            x=KL.Concatenate()([x1,x2,x3,x4,input])
            x=KL.Conv2D(filter,(1,1))(x)
            return x

        def combine(input1,input2,filter):
            x=KL.Activation('relu')(input1)
            x=KL.Concatenate()([x,input2])
            x=KL.Conv2D(filter,(1,1))(x)
            return x
        # inputs=KM.Input(shape=(self.config.IMAGE_H, self.config.IMAGE_W, self.config.IMAGE_C))
        #KB.set_image_dim_ordering('th')
        inputs=KE.Input(shape=(self.img_height, self.img_width, self.img_channel))

        # Encoder
        c1=x=KL.Conv2D(32,(1,1),strides=(1,1),dilation_rate=1, padding='same')(inputs)
        print(x.shape)
        c2=x=ResBlock(x,32,(3,3),[1,3,15,31],(1,1))
        print(x.shape)
        if (self.img_height, self.img_width) >= (64, 64):
            x=KL.Conv2D(64,(1,1),strides=(2,2), padding='same')(x)
            c3=x=ResBlock(x,64,(3,3),[1,3,15,31],(1,1))
            print(x.shape)
            N = 64*2
        if (self.img_height, self.img_width) >= (128, 128):
            x=KL.Conv2D(128,(1,1),strides=(2,2), padding='same')(x)
            c4=x=ResBlock(x,128,(3,3),[1,3,15],(1,1))
            print(x.shape)
            print('aqui'*20)
            N = 128*2
        if (self.img_height, self.img_width) >= (256, 128):
            x=KL.Conv2D(256,(1,1),strides=(2,2), padding='same')(x)
            c5=x=ResBlock(x,256,(3,3),[1,3,15],(1,1))
            print(x.shape)
            N = 256*2
        if (self.img_height, self.img_width) >= (512, 512):
            x=KL.Conv2D(512,(1,1),strides=(2,2), padding='same')(x)
            c6=x=ResBlock(x,512,(3,3),[1],(1,1))
            print(x.shape)
            N = 512*2

        # Talvez isso deva ser sempre 1024
        N = 1024
        x=KL.Conv2D(N,(1,1),strides=(2,2), padding='same')(x)
        x=ResBlock(x,N,(3,3),[1],(1,1))

        print('[DEBUG]'*10)
        print(x.shape)

        x=PSPPooling(x,N)

        # Decoder
        if (self.img_height, self.img_width) >= (512, 512):
            x=KL.Conv2D(512,(1,1))(x)
            x=KL.UpSampling2D()(x)
            x=combine(x,c6,512)
            x=ResBlock(x,512,(3,3),[1],1)

        if (self.img_height, self.img_width) >= (256, 256):
            x=KL.Conv2D(256,(1,1))(x)
            x=KL.UpSampling2D()(x)
            x=combine(x,c5,256)
            x=ResBlock(x,256,(3,3),[1,3,15],1)

        if (self.img_height, self.img_width) >= (128, 128):
            x=KL.Conv2D(128,(1,1))(x)
            x=KL.UpSampling2D()(x)
            x=combine(x,c4,128)
            x=ResBlock(x,128,(3,3),[1,3,15],1)

        if (self.img_height, self.img_width) >= (64, 64):
            x=KL.Conv2D(64,(1,1))(x)
            x=KL.UpSampling2D()(x)
            x=combine(x,c3,64)
            x=ResBlock(x,64,(3,3),[1,3,15,31],1)

        x=KL.Conv2D(32,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c2,32)

        x=ResBlock(x,32,(3,3),[1,3,15,31],1)
        x=combine(x,c1,32)

        x=PSPPooling(x,32)
        x=KL.Conv2D(self.config.CLASSES_NUM,(1,1))(x)
        x=KL.Activation('softmax')(x)

        model=KM.Model(inputs=inputs,outputs=x)
        # Talvez mudar para Adam
        adam = Adam(lr = 0.001 , beta_1=0.9)
        # model.compile(optimizer=SGD(lr=0.001,momentum=0.8),loss=Tanimoto_loss,metrics=['accuracy'])
        model.compile(optimizer=adam,loss=Tanimoto_loss,metrics=['accuracy'])
        model.summary()
        return model

    def loadWeight(self,path):
        self.model.load_weights(path)

    def predict(self,img):
        img=img-self.config.MEAN
        img=np.expand_dims(img,axis=0)
        img=self.model.predict(img)
        img=img[0]
        result=np.argmax(img,axis=-1)
        return result
