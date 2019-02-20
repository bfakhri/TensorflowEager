# Inspired by the TF Tutorial: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

# Dataset
import tensorflow_datasets as tfds

# Constants to eventually parameterise
LOGDIR = './logs'

# Activation function to use for layers
#act_func = tf.nn.softplus
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
    ''' Performs a 2d convolution '''
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

summary_writer = tf.contrib.summary.create_file_writer(LOGDIR, flush_millis=100)
summary_writer.set_as_default()
global_step = tf.train.get_or_create_global_step()


class Model:
    ' Simple Image Classification Model (defined by CNN) ' 

    def __init__(self, input_shape, output_size, learn_rate=1e-4):
        ' Initializes model parameters and optimizer ' 

        # Stores model params
        self.vars = []
        ## Conv Layers
        #self.ds_chans = input_shape[2]
        self.input_shape = input_shape
        self.output_size = output_size
        next_size = int((input_shape[0]+1)/2)
        fn_out = 32*self.input_shape[2]
        self.conv1_w = weight_variable([5, 5, self.input_shape[2], fn_out], name='W')
        self.conv1_b = bias_variable([fn_out], name='B')
        self.vars.append(self.conv1_w)
        self.vars.append(self.conv1_b)
        fn_in = fn_out
        fn_out = 2*fn_in 
        next_size = int((next_size+1)/2)
        self.conv2_w = weight_variable([5, 5, fn_in, fn_out], name='W')
        self.conv2_b = bias_variable([fn_out], name='B')
        self.vars.append(self.conv2_w)
        self.vars.append(self.conv2_b)
        ## FC Layers
        self.FC1_Size = 512 
        
        self.W_fc_up1 = weight_variable([next_size*next_size*fn_out, self.FC1_Size])
        self.b_fc_up1 = bias_variable([self.FC1_Size])
        self.vars.append(self.W_fc_up1)
        self.vars.append(self.b_fc_up1)
        self.W_fc_up2 = weight_variable([self.FC1_Size, output_size])
        self.b_fc_up2 = bias_variable([output_size])
        self.vars.append(self.W_fc_up2)
        self.vars.append(self.b_fc_up2)

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

            with tf.name_scope('FC1'):
                h = act_func(tf.matmul(z, self.W_fc_up1) + self.b_fc_up1)
            with tf.name_scope('FC2'):
                predictions = tf.nn.softmax(tf.matmul(h, self.W_fc_up2) + self.b_fc_up2)
            return predictions 

        
    def learn(self, x_input, labels):
        ' Learns from the batch ' 

        # Track gradients
        with tf.GradientTape() as tape:
            output = self.crunch(x_input)
            with tf.name_scope('Generation_Loss'):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
                tf.contrib.summary.scalar('Cross Entropy', cross_entropy)

                grads = tape.gradient(cross_entropy, self.vars)
                self.optimizer.apply_gradients(zip(grads, self.vars))
                global_step.assign_add(1)
                return output, cross_entropy 

    def calc_accuracy(self, predictions, labels, is_val=False):
        ' Calculates accuracy as a percentage and records it for TB '

        correct_prediction = tf.equal(tf.argmax(labels,1), tf.argmax(predictions,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        if(is_val):
            tf.contrib.summary.scalar('Validation Acc', accuracy)
        else:
            tf.contrib.summary.scalar('Training Acc', accuracy)
        return accuracy  

    def validate(self, x_input, labels):
        ' Performs an accuracy assessment without learning from the batch ' 
        
        output = self.crunch(x_input)
        val_acc = self.calc_accuracy(output, labels, is_val=True)
        return val_acc

# Get Data
# Construct a tf.data.Dataset
#ds_name = 'mnist'
#ds_name = 'cifar10'
ds_name = 'cifar100'
#ds_name = 'omniglot'
(ds_train, ds_test), ds_info = tfds.load(name=ds_name, split=['train', 'test'], with_info=True)
img_shape = tf.TensorShape(ds_info.features['image'].shape)
label_shape = tf.TensorShape(ds_info.features['label'].shape)
num_classes = ds_info.features['label'].num_classes
print('DS Shape: ')
print(img_shape, num_classes)


# Creates a classifier model
model = Model(img_shape.as_list(), num_classes)

# Preparing datasets (training and validation)
# Batch size of 1024 the repeats when iterated through
ds_train = ds_train.batch(256).repeat()
ds_test = ds_test.batch(256).repeat()

# Converts validation set into an iterator so we can iterate through it
ds_test_iter = iter(ds_test)

# Perform the training loop (forever)
for idx,batch in enumerate(ds_train):
    # Prepare training inputs
    x_inputs = tf.math.divide(tf.cast(batch['image'], tf.float32), tf.constant(255.0, dtype=tf.float32))
    y_labels = tf.one_hot(batch['label'], depth=num_classes)
    # Prepare validation inputs
    val_batch = next(ds_test_iter)
    val_x_inputs = tf.math.divide(tf.cast(val_batch['image'], tf.float32), tf.constant(255.0, dtype=tf.float32))
    val_y_labels = tf.one_hot(val_batch['label'], depth=num_classes)
    # Train and validate
    with tf.contrib.summary.record_summaries_every_n_global_steps(100):
        preds, loss = model.learn(x_inputs, y_labels) 
        train_acc = model.calc_accuracy(preds, y_labels)
        val_acc = model.validate(val_x_inputs, val_y_labels)
        print('idx: ', idx, 'Loss: ', loss.numpy(), 'TrainAcc: ', train_acc.numpy()*100, '\tValAcc: ', val_acc.numpy()*100)

