import os

from keras.models import *
from keras.layers import *
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

from keras import backend as K





# define the path to load the VGG16 weights
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG16_Weights_path = file_path + "/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
# VGG19_Weights_path = file_path + "/vgg19_weights_tf_dim_ordering_tf_kernels.h5"


#VGG16_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/tag/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
#VGG19_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/tag/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'



def VGG16(input_height=224, input_width=224, fc_end_layer = 'fc1'):
    
    
    min_size = 32
    
    assert (input_height > min_size or input_width > min_size), 'The input images are smaller than the minimum size' 
    
    img_input = Input(shape=(input_height,input_width,3))
    
    
    
    # Main VGG16 Architecture 
    
     # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

   
    x = Flatten(name='flatten')(x)
    if fc_end_layer == 'fc1':        
        x = Dense(4096, activation='relu', name='fc1')(x)
    else:
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        
        
   

    # Create model.
    model = Model(img_input, x, name='vgg16')

    
    # VGG16_Weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', VGG16_WEIGHTS_PATH, cache_subdir='models')
    model.load_weights(VGG16_Weights_path, by_name=True)
    

    
    return model
    
def VGG19(input_height=224, input_width=224, fc_end_layer = 'fc1'):
    
    min_size = 32
    
    assert (input_height > min_size or input_width > min_size), 'The input images are smaller than the minimum size' 
    
    img_input = Input(shape=(input_height,input_width,3))
    
    
    
    # Main VGG19 Architecture 
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    
    
    x = Flatten(name='flatten')(x)
    if fc_end_layer == 'fc1':        
        x = Dense(4096, activation='relu', name='fc1')(x)
    else:
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        
        
   

    # Create model.
    model = Model(img_input, x, name='vgg19')
    
    # VGG19_Weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', VGG19_WEIGHTS_PATH, cache_subdir='models')

    model.load_weights(VGG19_Weights_path, by_name=True)
    
    print('The VGG19 model is built')
    
    
    return model        
        
        
        
        
        
        
        
        
        
    
    