import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

# Dataset
import tensorflow_datasets as tfds

# Constants to eventually parameterise
LOGDIR = './logs/autoencoder_gg/'

# Activation function to use for layers
act_func = tf.nn.tanh

# Enable or disable GPU
SESS_CONFIG = tf.ConfigProto(device_count = {'GPU': 1})


class Model:
    ' Simple Image Classification Model (defined by CNN) ' 

    def __init__(self, input_shape, num_layers=4, activation=tf.nn.relu, layer_width=64, bottleneck_chans = 4, learn_rate=1e-4):
        ' Initializes model parameters and optimizer ' 
        ' Assumes an input shape of [height, width, channels]'

        # Stores model params
        self.vars = []
        self.layers = []
        self.input_shape = input_shape
        self.shape_list = [] 
        self.shape_list.append(input_shape)
        
        # Down-sampling Layers
        for l in range(num_layers):
            # First layer
            if(l == 0):
                in_chans = input_shape[2]
                out_chans = layer_width
                cur_shape = [1,] + input_shape
            # Last Layer
            elif(l == num_layers-1):
                in_chans = out_chans
                out_chans = bottleneck_chans 
            # Middle layers
            else:
                in_chans = out_chans
                out_chans = layer_width 
            
            f_height = 5
            f_width = 5

            layer = tf.layers.Conv2D(out_chans, (f_height, f_width), strides=[1,1], padding='valid', activation=activation, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.random_normal, name='Conv'+str(l))
            layer.build(cur_shape)
            cur_shape = layer.compute_output_shape(cur_shape)
            self.shape_list.append(cur_shape)
            self.layers.append(layer)

        # Up-sampling Layers
        for l in range(num_layers):
            # First layer
            if(l == 0):
                in_chans = bottleneck_chans
                out_chans = layer_width
            # Last Layer
            elif(l == num_layers-1):
                in_chans = out_chans
                out_chans = input_shape[2]
            # Middle layers
            else:
                in_chans = out_chans
                out_chans = layer_width 
            
            f_height = 5
            f_width = 5

            layer = tf.layers.Conv2DTranspose(out_chans, (f_height, f_width), strides=[1,1], padding='valid', activation=activation, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.random_normal, name='ConvTP'+str(l))
            layer.build(cur_shape)
            cur_shape = layer.compute_output_shape(cur_shape)
            self.shape_list.append(cur_shape)
            self.layers.append(layer)

        # Our Optimizer
        self.optimizer = tf.train.AdamOptimizer(learn_rate)

        # Grab all variables
        for l in self.layers:
            self.vars.extend(l.weights) 
        for idx,shape in enumerate(self.shape_list):
            if(idx == 0):
                out_shape = None
            else:
                out_shape = self.layers[idx-1].weights[0].shape

            print('Layer: ', str(idx), shape, 'Weights: ', out_shape)


    def crunch(self, x_input):
        ' Generates outputs (predictions) from inputs to the model '

        with tf.name_scope('MainGraph'):
            for l in range(len(self.layers)):
                if(l == 0):
                    h = self.layers[0](x_input)
                    tf.contrib.summary.image(self.layers[l].name, h[:,:,:,:3], max_images=1) 
                else:
                    h = self.layers[l](h)
                    tf.contrib.summary.image(self.layers[l].name, h[:,:,:,:3], max_images=1) 
            
            #x_hat = tf.sigmoid(h)
            #x_hat = h
            x_hat = tf.sigmoid(tf.image.per_image_standardization(h))
            return x_hat

        
    def learn(self, x_input):
        ' Learns from the batch ' 

        # Track gradients
        with tf.GradientTape() as tape:
            tape.watch(x_input)
            output = self.crunch(x_input)
            tf.contrib.summary.image('Reconstructed Image', output, max_images=3)
            with tf.name_scope('Generation_Loss'):
                reconstruction_loss = tf.losses.mean_squared_error(labels=x_input, predictions=output)
                tf.contrib.summary.scalar('Recon Loss', reconstruction_loss)

                grads = tape.gradient(reconstruction_loss, self.vars)
                self.optimizer.apply_gradients(zip(grads, self.vars))
                #self.optimizer.apply_gradients(zip(grads, self.layers[0].weights))
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

        for l in self.layers:
            tf.contrib.summary.histogram('Weights_'+l.name, l.weights[0])
            tf.contrib.summary.histogram('Biases_'+l.name, l.weights[1])

# Get Data
# Construct a tf.data.Dataset
#ds_name = 'mnist'
#ds_name = 'cifar10'
#ds_name = 'cifar100'
#ds_name = 'omniglot'
ds_name = 'celeb_a'
#ds_name = 'fashion_mnist'
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
ds_train = ds_train.batch(64).repeat()
ds_test = ds_test.batch(64).repeat()

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
    with tf.contrib.summary.record_summaries_every_n_global_steps(10):
        preds, loss = model.learn(x_inputs) 
        print('idx: ', idx, 'Loss: ', loss.numpy())
        model.validate(val_x_inputs)

