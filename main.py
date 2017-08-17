import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    with sess.as_default():
		vgg_tag = 'vgg16'
		
		#   Use tf.saved_model.loader.load to load the model and weights
		model_vgg = tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
		
		vgg_input_tensor_name      = tf.get_default_graph().get_tensor_by_name('image_input:0')
		vgg_keep_prob_tensor_name  = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
		vgg_layer3_out_tensor_name = tf.get_default_graph().get_tensor_by_name('layer3_out:0')
		vgg_layer4_out_tensor_name = tf.get_default_graph().get_tensor_by_name('layer4_out:0')
		vgg_layer7_out_tensor_name = tf.get_default_graph().get_tensor_by_name('layer7_out:0')
    
    return vgg_input_tensor_name, vgg_keep_prob_tensor_name, vgg_layer3_out_tensor_name, vgg_layer4_out_tensor_name, vgg_layer7_out_tensor_name
	
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
	
	# Build Fully Convolutional Network (FCN)
	
	###########VGG is "Encoder" portion of FCN##########
    
	#1x1 Convolutions applied to VGG layers
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'SAME',  kernel_initializer=tf.truncated_normal_initializer(stddev=0.001))
	layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding = 'SAME',  kernel_initializer=tf.truncated_normal_initializer(stddev=0.001))
	layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding = 'SAME',  kernel_initializer=tf.truncated_normal_initializer(stddev=0.001))
	
	###########Decoder###########
		
    # Transposed Convolutional Layer applied to (VGG Layer 7 + 1x1 Conv): Upsample, kernel (4 x 4) and Stride (2 x 2)
    layer7_tconv = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4, strides = (2, 2), padding = 'SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001))

    # Apply Skip layer connecting transposed convolutional layer 7 and (VGG Layer 4 + 1x1 Conv)
    skip_layer_l7t_l4 = tf.add(layer4_conv_1x1,layer7_tconv)

    # Transposed Convolutional Layer applied to skip layer: Upsample, kernel (4 x 4) and Stride (2 x 2)
    layer4_tconv = tf.layers.conv2d_transpose(skip_layer1, num_classes, 4, strides = (2, 2), padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001))
	
	# Apply Skip layer connecting transposed convolutional layer 4 and (VGG Layer 3 + 1x1 Conv)
    skip_layer_l4t_l3 = tf.add(layer3_conv_1x1, layer4_tconv)

    # Transposed Convolutional Layer applied to skip layer: Upsample, kernel (16 x 16) and Stride (8 x 8)
    output_layer = tf.layers.conv2d_transpose(skip_layer2, num_classes, 16, strides = (8, 8), padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.001))

    return output_layer
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
	# Set size of logits
	logits = tf.reshape(nn_last_layer, (-1, num_classes))
	# Set size of labels
	labels = tf.reshape(correct_label, (-1, num_classes))
	# Compute cross entropy loss
	softmax_function   = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
	cross_entropy_loss = tf.reduce_mean(softmax_function)
	# Setup training operations
	train_op           = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss);
	
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
	#Hyper parameters
	keep_probability = 0.65 #0.70 # 0.5
	learn_rate       = 1e-3 #1e-4 # 0.001
	
    for epoch_idx in range(epochs):
		for img_idx, img_lbl_idx in get_batches_fn(batch_size):
			_,loss = sess.run([train_op,cross_entropy_loss], feed_dict = {input_image: img_idx, correct_label: img_lbl_idx, keep_prob: keep_probability, learning_rate:learn_rate})
            print("Current Epoch: {} ".format(epoch_idx+1), "/{} -".format(epochs))
			print("Loss: {:.6f}".format(loss))
tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
	
	#Hyper Parameters
	epochs     = 37 #45 #10
	batch_size = 17 #20 #16

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    # https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
		input_image, keep_prob, vgg_layer_3, vgg_layer_4, vgg_layer_7 = load_vgg(sess, vgg_path)
		
		decoder_last_layer = layers(vgg_layer_3, vgg_layer_4, vgg_layer_7, num_classes)
		
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
		
        learning_rate = tf.placeholder(dtype=tf.float32)
		
		logits, train_op, cross_entropy_loss = optimize(decoder_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
		sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, training_operation, cross_entropy_loss, image_input,correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
		helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
		saver = tf.train.Saver()
        saver.save(sess, 'FCN_Model')
        print("FCN Model Successfully Saved")
		
        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
