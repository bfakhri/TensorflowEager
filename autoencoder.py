# Inspired by the TF Tutorial: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

# Dataset
import tensorflow_datasets as tfds

# Constants to eventually parameterise
LOGDIR = './logs/autoencoder/'

# Activation function to use for layers
act_func = tf.nn.tanh

# Enable or disable GPU
SESS_CONFIG = tf.ConfigProto(device_count = {'GPU': 1})

# Define variable functions
def weight_variable(shape, name='W'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name='B'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, name='conv'):
    ' Performs a 2d convolution '
    with tf.name_scope(name):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max-Pooling Function - Pooling explained here: 
# http://ufldl.stanford.edu/tutorial/supervised/Pooling/
def max_pool_2x2(x, name='max_pool'):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Define a Convolutional Layer 
def conv_layer(x, fan_in, fan_out, name='convl'):
    with tf.name_scope(name):
        # Create Weight Variables
        W = weight_variable([5, 5, fan_in, fan_out], name='W')
        B = bias_variable([fan_out], name='B')
        # Convolve the input using the weights
        conv = conv2d(x, W)
        # Push input+bias through activation function
        activ = tf.nn.relu(conv + B)
        # Create histograms for visualization
        tf.contrib.summary.histogram('Weights', W)
        tf.contrib.summary.histogram('Biases', B)
        tf.contrib.summary.histogram('Activations', activ) 
        # MaxPool Output
        return max_pool_2x2(activ)

def conver(x, w, b, name='convll'):
    with tf.name_scope(name):
        # Convolve the input using the weights
        conv = conv2d(x, w)
        # Push input+bias through activation function
        activ = tf.nn.relu(conv + b)
        # Create histograms for visualization
        tf.contrib.summary.histogram('Weights', w)
        tf.contrib.summary.histogram('Biases', b)
        tf.contrib.summary.histogram('Activations', activ) 
        # MaxPool Output
        return max_pool_2x2(activ)



class Model:
    ' Simple Image Classification Model (defined by CNN) ' 

    def __init__(self, input_shape, learn_rate=1e-4):
        ' Initializes model parameters and optimizer ' 

        # Stores model params
        self.vars = []

        ## Conv Layers
        self.input_shape = input_shape
        next_size = int((input_shape[0]+1)/2)
        #fn_out = 32*self.input_shape[2]
        fn_out = 4*self.input_shape[2]
        self.conv1_w = weight_variable([5, 5, self.input_shape[2], fn_out], name='W')
        self.conv1_b = bias_variable([fn_out], name='B')
        self.vars.append(self.conv1_w)
        self.vars.append(self.conv1_b)
        fn_in = fn_out
        #fn_out = 2*fn_in 
        fn_out = fn_in 
        next_size = int((next_size+1)/2)
        self.conv2_w = weight_variable([5, 5, fn_in, fn_out], name='W')
        self.conv2_b = bias_variable([fn_out], name='B')
        self.vars.append(self.conv2_w)
        self.vars.append(self.conv2_b)

        ## FC Upscaling Layers
        self.FC1_Size = tf.TensorShape(input_shape).num_elements()
        self.W_fc_up1 = weight_variable([next_size*next_size*fn_out, self.FC1_Size])
        self.b_fc_up1 = bias_variable([self.FC1_Size])
        self.vars.append(self.W_fc_up1)
        self.vars.append(self.b_fc_up1)

        # Our Optimizer
        self.optimizer = tf.train.AdamOptimizer(learn_rate)


    def crunch(self, x_input):
        ' Generates outputs (predictions) from inputs to the model '

        with tf.name_scope('MainGraph'):
            with tf.name_scope('Inputs'):
                # Reshape X to make it into a 2D image
                x_image = tf.reshape(x_input, [-1,]+self.input_shape)
                tf.contrib.summary.image('original_image', x_image, max_images=3)
            # FC Encoder Layers
            with tf.name_scope('Convs'):
                conv1 = conver(x_image, self.conv1_w, self.conv1_b, name='Conv1') 
                conv2 = conver(conv1, self.conv2_w, self.conv2_b, name='Conv2') 
                conv_sig = tf.sigmoid(conv2)

            with tf.name_scope('encoder_latent'):
                z = tf.layers.flatten(conv_sig)
                tf.contrib.summary.histogram('latent_z', z)

            with tf.name_scope('FC2'):
                x_hat = tf.nn.sigmoid(tf.matmul(z, self.W_fc_up1) + self.b_fc_up1)

            return x_hat 

        
    def learn(self, x_input):
        ' Learns from the batch ' 

        # Track gradients
        with tf.GradientTape() as tape:
            output = self.crunch(x_input)
            output_img = tf.reshape(output, [-1]+self.input_shape)
            tf.contrib.summary.image('Reconstructed Image', output_img, max_images=3)
            with tf.name_scope('Generation_Loss'):
                reconstruction_loss = tf.losses.mean_squared_error(labels=x_input, predictions=output_img)
                tf.contrib.summary.scalar('Recon Loss', reconstruction_loss)

                grads = tape.gradient(reconstruction_loss, self.vars)
                self.optimizer.apply_gradients(zip(grads, self.vars))
                global_step.assign_add(1)
                return output, reconstruction_loss 

    def validate(self, x_input):
        ' Takes an image from the validation set and produces an output from it ' 
        
        output = self.crunch(x_input)  
        output_rs = tf.reshape(output, [-1, self.input_shape[0]*self.input_shape[1], self.input_shape[2]])
        x_input_rs = tf.reshape(x_input, [-1, self.input_shape[0]*self.input_shape[1], self.input_shape[2]])
        # Get last three of each
        concat = tf.concat([x_input_rs, output_rs], axis=1)
        concat_img = tf.reshape(concat, [-1, self.input_shape[0]*2, self.input_shape[1], self.input_shape[2]])
        tf.contrib.summary.image('Validation Pair', concat_img, max_images=3)

# Get Data
# Construct a tf.data.Dataset
ds_name = 'mnist'
#ds_name = 'cifar10'
#ds_name = 'cifar100'
#ds_name = 'omniglot'
#ds_name = 'celeb_a'
(ds_train, ds_test), ds_info = tfds.load(name=ds_name, split=['train', 'test'], with_info=True)
img_shape = tf.TensorShape(ds_info.features['image'].shape)
print('DS Shape: ')
print(img_shape)

summary_writer = tf.contrib.summary.create_file_writer(LOGDIR+ds_name, flush_millis=100)
summary_writer.set_as_default()
global_step = tf.train.get_or_create_global_step()

# Creates a classifier model
model = Model(img_shape.as_list())

# Preparing datasets (training and validation)
# Batch size of 1024 the repeats when iterated through
ds_train = ds_train.batch(4).repeat()
ds_test = ds_test.batch(4).repeat()

# Converts validation set into an iterator so we can iterate through it
ds_test_iter = iter(ds_test)

# Perform the training loop (forever)
for idx,batch in enumerate(ds_train):
    # Prepare training inputs
    x_inputs = tf.math.divide(tf.cast(batch['image'], tf.float32), tf.constant(255.0, dtype=tf.float32))
    # Prepare validation inputs
    val_batch = next(ds_test_iter)
    val_x_inputs = tf.math.divide(tf.cast(val_batch['image'], tf.float32), tf.constant(255.0, dtype=tf.float32))
    # Train and validate
    with tf.contrib.summary.record_summaries_every_n_global_steps(100):
        preds, loss = model.learn(x_inputs) 
        print('idx: ', idx, 'Loss: ', loss.numpy())
        model.validate(val_x_inputs)

