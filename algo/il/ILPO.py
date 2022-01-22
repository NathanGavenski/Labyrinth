"""ILPO network for images."""
import collections
from collections import deque
import glob
import math
from multiprocessing.sharedctypes import Value
import os
import random
import time
from os import listdir
from os.path import isfile, join
from tkinter import W

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorboard_wrapper.tensorboard import Tensorboard as Board

CROP_SIZE = 128

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, expectation, min_output, actions, gen_loss_L1, gen_grads_and_vars, train")

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        print(filter)
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch = tf.shape(batch_input)[0]
        in_height = int(batch_input.shape[1])
        in_width = int(batch_input.shape[2])
        in_channels = int(batch_input.shape[3])
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def fully_connected(inputs, n_outputs, reuse=False, scope=None):
    inputs = slim.flatten(inputs)

    return slim.fully_connected(inputs, n_outputs, activation_fn=None, reuse=reuse, scope=scope)

    with tf.variable_scope("fc"):
        w_fc = weight_variable([int(inputs.shape[-1]), n_outputs])
        b_fc = bias_variable([n_outputs])
        return tf.matmul(inputs, w_fc) + b_fc

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

class ImageILPO:
    def __init__(
        self, 
        input_dir, 
        output_dir,
        checkpoint_dir,
        batch_size,
        trace_freq=0,
        summary_freq=100,
        save_freq=5000,
        display_freq=0,
        progress_freq=50,
        max_epochs=5,
        max_steps=0,
        flip=True, 
        direction='AtoB', 
        ngf=128,
        n_actions=4,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.checkpoint = checkpoint_dir

        self.flip = flip
        self.which_direction = direction
        self.batch_size = batch_size
        self.ngf = ngf
        self.max_epochs = max_epochs
        self.max_steps = 10000
        self.n_actions = n_actions

        self.trace_freq = trace_freq
        self.summary_freq = summary_freq
        self.save_freq = save_freq
        self.display_freq = display_freq
        self.progress_freq = progress_freq

    def process_inputs(self, inputs):
        inputs = inputs / 255.
        inputs = tf.image.resize_images(inputs, [CROP_SIZE, CROP_SIZE])
        inputs = preprocess(inputs)
        return inputs

    def load_examples(self):
        if not os.path.exists(self.input_dir):
            raise Exception("input_dir does not exist")

        input_paths = glob.glob(os.path.join(self.input_dir, "*.png"))
        decode = tf.image.decode_png

        if len(input_paths) == 0:
            raise Exception("input_dir contains no image files")

        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

        # if the image names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        if all(get_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        else:
            input_paths = sorted(input_paths)

        with tf.name_scope("load_images"):
            path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
            reader = tf.WholeFileReader()
            paths, contents = reader.read(path_queue)
            raw_input = decode(contents, channels=3)
            raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

            assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
            with tf.control_dependencies([assertion]):
                raw_input = tf.identity(raw_input)

            raw_input = tf.image.resize_images(raw_input, [CROP_SIZE, CROP_SIZE*2])
            raw_input.set_shape([CROP_SIZE, CROP_SIZE*2, 3])

            # break apart image pair and move to range [-1, 1]
            width = CROP_SIZE
            a_images = preprocess(raw_input[:,0:width,:])
            b_images = preprocess(raw_input[:,width:width*2,:])

        if self.which_direction == "AtoB":
            inputs, targets = [a_images, b_images]
        elif self.which_direction == "BtoA":
            inputs, targets = [b_images, a_images]
        else:
            raise Exception("invalid direction")

        # synchronize seed for image operations so that we do the same operations to both
        # input and output images
        seed = random.randint(0, 2**31 - 1)
        def transform(image):
            r = image
            if self.flip:
                r = tf.image.random_flip_left_right(r, seed=seed)

            r.set_shape([CROP_SIZE, CROP_SIZE, 3])

            return r

        with tf.name_scope("input_images"):
            input_images = transform(inputs)

        with tf.name_scope("target_images"):
            target_images = transform(targets)

        paths_batch, inputs_batch, targets_batch = tf.train.batch(
            [paths, input_images, target_images],
            batch_size=self.batch_size,
            num_threads=1,
            capacity=500000)

        steps_per_epoch = int(math.ceil(len(input_paths) / self.batch_size))

        return Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )

    def create_encoder(self, state):
        """Creates state embedding."""

        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = conv(state, self.ngf, stride=2)
            layers.append(output)

        layer_specs = [
            self.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(rectified, out_channels, stride=2)
                layers.append(convolved)

        return layers

    def create_generator(self, layers, generator_outputs_channels):
        """Returns next state prediction given a combined state and latent action."""

        s_t_layers = list(layers)
        layer_specs = [
            (self.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (self.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (self.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (self.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (self.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (self.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(s_t_layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = s_t_layers[-1]
                else:
                    input = tf.concat([s_t_layers[-1], s_t_layers[skip_layer]], axis=3)

                rectified = lrelu(input, .2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(rectified, out_channels)

                s_t_layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([s_t_layers[-1], s_t_layers[0]], axis=3)
            rectified = lrelu(input, .2)
            output = deconv(rectified, generator_outputs_channels)
            s_t_layers.append(output)

        return s_t_layers[-1]

    def create_ilpo(self, s_t, generator_outputs_channels):
        """Creates ILPO network."""

        # Create state embeddings.
        with tf.variable_scope("state_encoding"):
            s_t_layers = self.create_encoder(s_t)

        # Predict latent action probabilities.
        with tf.variable_scope("action"):
            flat_s = lrelu(s_t_layers[-1], .2)
            action_prediction = fully_connected(flat_s, self.n_actions)

            for a in range(self.n_actions):
                tf.summary.histogram("action_{}".format(a), action_prediction[:,a])
            tf.summary.histogram("action_max", tf.nn.softmax(action_prediction))

            action_prediction = tf.nn.softmax(action_prediction)

        # predict next state from latent action and current state.
        outputs = []
        shape = [ind for ind in flat_s.shape]
        shape[0] = tf.shape(flat_s)[0]

        for a in range(self.n_actions):
            # there is one generator g(s,z) that takes in a state s and latent action z.
            with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                action = tf.one_hot([a], self.n_actions)

                # obtain fully connected latent action to concatenate with state.
                action = fully_connected(action, int(flat_s.shape[-1]), reuse=tf.AUTO_REUSE, scope="action_embedding")
                action = lrelu(action, .2)

                # tile latent action embedding.
                action = tf.tile(action, [1, tf.shape(s_t)[0]])
                action = tf.reshape(action, shape)

                # concatenate state and action.
                state_action = slim.flatten(tf.concat([flat_s, action], axis=-1))
                state_action = fully_connected(state_action, int(flat_s.shape[-1]), reuse=tf.AUTO_REUSE, scope="state_action_embedding")
                state_action = tf.reshape(state_action, tf.shape(flat_s))

                s_t_layers[-1] = state_action
                outputs.append(self.create_generator(s_t_layers, generator_outputs_channels))

        expected_states = 0
        shape = tf.shape(outputs[0])

        # compute expected next state as sum_z p(z|s)*g(s,z)
        for a in range(self.n_actions):
            expected_states += tf.multiply(tf.stop_gradient(slim.flatten(outputs[a])), tf.expand_dims(action_prediction[:, a], -1))
        expected_states = tf.reshape(expected_states, shape)

        return (expected_states, outputs, action_prediction)

    def create_model(self, inputs, targets):
        """ Initializes ILPO model and losses."""

        global_step = tf.contrib.framework.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        with tf.variable_scope("ilpo_loss"):
            out_channels = int(targets.get_shape()[-1])
            expected_outputs, outputs, actions = self.create_ilpo(inputs, out_channels)

            # compute loss on expected next state.
            delta = slim.flatten(targets - inputs)
            gen_loss_exp = tf.reduce_mean(
                tf.reduce_sum(tf.losses.mean_squared_error(delta, slim.flatten(expected_outputs),
                                                   reduction=tf.losses.Reduction.NONE), axis=1))

            # compute loss on min next state.
            all_loss = []

            for out in outputs:
                all_loss.append(tf.reduce_sum(
                    tf.losses.mean_squared_error(delta, slim.flatten(out),
                    reduction=tf.losses.Reduction.NONE),
                    axis=1))

            stacked_min_loss = tf.stack(all_loss, axis=-1)
            gen_loss_min = tf.reduce_mean(tf.reduce_min(stacked_min_loss, axis=1))

            gen_loss_L1 = gen_loss_exp + gen_loss_min

            # obtain images and scalars for summaries.
            tf.summary.scalar("expected_gen_loss", gen_loss_exp)
            tf.summary.scalar("min_gen_loss", gen_loss_min)

            min_index = tf.argmin(all_loss)
            min_index = tf.one_hot(min_index, self.n_actions)

            shape = tf.shape(out)
            min_img = tf.stack([slim.flatten(out) for out in outputs], axis=-1)
            min_img = tf.reduce_sum(tf.multiply(min_img, tf.expand_dims(min_index, 1)), -1)
            min_img = tf.reshape(min_img, shape)

        with tf.name_scope("ilpo_train"):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("ilpo")]
            gen_optim = tf.train.AdamOptimizer(0.0002, 0.5)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss_L1, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        return Model(
            gen_loss_L1=gen_loss_L1,
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=[inputs + out for out in outputs],
            expectation=inputs + expected_outputs,
            min_output=inputs + min_img,
            actions=actions,
            train=tf.group(gen_loss_L1, incr_global_step, gen_train),
        )

    def train_examples(self, examples):
        print("examples count = %d" % examples.count)

        # inputs and targets are [batch_size, height, width, channels]
        model = self.create_model(examples.inputs, examples.targets)

        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        expectation = deprocess(model.expectation)
        min_output = deprocess(model.min_output)

        def convert(image):
            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

        # reverse any processing on images so they can be written to disk or displayed to user
        with tf.name_scope("convert_inputs"):
            converted_inputs = convert(inputs)

        with tf.name_scope("convert_targets"):
            converted_targets = convert(targets)

        with tf.name_scope("convert_expectation"):
            converted_expectation = convert(expectation)

        with tf.name_scope("convert_min_output"):
            converted_min_output = convert(min_output)

        with tf.name_scope("encode_images"):
            display_fetches = {
                "paths": examples.paths,
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
                "expected_outputs": tf.map_fn(tf.image.encode_png, converted_expectation, dtype=tf.string, name="expected_output_pngs"),
                "min_outputs": tf.map_fn(tf.image.encode_png, converted_min_output, dtype=tf.string,
                                              name="min_output_pngs"),
            }

        # summaries
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", converted_inputs, 3)

        with tf.name_scope("targets_summary"):
            tf.summary.image("targets", converted_targets)

        with tf.name_scope("outputs_summary_expected"):
            tf.summary.image("outputs_expected", converted_expectation, 3)

        with tf.name_scope("outputs_summary_min"):
            tf.summary.image("outputs_min", converted_min_output, 3)

        for i in range(0, len(model.outputs)):
            with tf.name_scope("outputs_" + str(i)):
                tf.summary.image("outputs_" + str(i), model.outputs[i], 3)

        summaries = tf.summary.image("merged", tf.concat(
            [converted_inputs, converted_targets, converted_expectation, converted_min_output], 1), 3)

        tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        for grad, var in model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        saver = tf.train.Saver(max_to_keep=1)

        logdir = self.output_dir if (self.trace_freq > 0 or self.summary_freq > 0) else None
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        config = tf.ConfigProto(
            inter_op_parallelism_threads=4,
            intra_op_parallelism_threads=4,
        )
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            print("parameter_count =", sess.run(parameter_count))

            if self.checkpoint is not None:
                print("loading model from checkpoint")
                checkpoint = tf.train.latest_checkpoint(self.checkpoint)
                try:
                    saver.restore(sess, checkpoint)
                except ValueError:
                    pass

            max_steps = 2 ** 32
            if self.max_epochs is not None:
                max_steps = examples.steps_per_epoch * self.max_epochs
            if self.max_steps is not None:
                max_steps = self.max_steps


            summary = sess.run(summaries)
            sv.summary_writer.add_summary(summary)

            start = time.time()

            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            for step in range(max_steps):

                options = None
                run_metadata = None
                if should(self.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(self.progress_freq):
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(self.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(self.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(self.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(self.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(self.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * self.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * self.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(self.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(self.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break

    def run(self, mode='train', seed=None, ):
        """Runs training method."""

        if tf.__version__.split('.')[0] != "1":
            raise Exception("Tensorflow version 1 required")

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if mode == "test" or mode == "export":
            if self.checkpoint is None:
                raise Exception("checkpoint required for test mode")

            # disable these features in test mode
            self.flip = False

        examples = self.load_examples()
        self.train_examples(examples)

class Policy(ImageILPO):
    def __init__(
        self,
        sess,
        shape,
        checkpoint,
        game,
        maze_path,
        ngf=128,
        batch_size=4,
        n_actions=4,
        real_actions=4,
        policy_lr=0.001,
        verbose=True,
        use_encoding=False,
        experiment=False,
        exp_writer=None,
        name=None,
    ):
        """Initializes the ILPO policy network."""

        mypath = f'{maze_path}train/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        self.mazes = np.array(mazes)

        mypath = f'{maze_path}eval/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        self.eval_mazes = np.array(mazes)

        self.sess = sess
        self.verbose = verbose
        self.use_encoding = use_encoding
        self.inputs = tf.placeholder("float", shape)
        self.targets = tf.placeholder("float", shape)
        self.state = tf.placeholder("float", shape)
        self.action = tf.placeholder("int32", [None])
        self.fake_action = tf.placeholder("int32", [None])
        self.reward = tf.placeholder("float", [None])
        self.exp_writer = exp_writer
        self.experiment = experiment
        self.policy_lr = policy_lr
        self.checkpoint = checkpoint
        self.n_actions = n_actions
        self.real_actions = real_actions
        self.game = game
        self.batch_size = batch_size
        self.ngf = ngf

        processed_inputs = self.process_inputs(self.inputs)
        processed_targets = self.process_inputs(self.targets)
        processed_state = self.process_inputs(self.state)

        self.model = self.create_model(processed_inputs, processed_targets)

        self.state_encoding = self.encode(processed_inputs)[-1]
        self.action_label, self.loss = self.action_remap_net(processed_state, self.action, self.fake_action)
        self.loss_summary = tf.summary.scalar("policy_loss", tf.squeeze(self.loss))

        if not self.experiment:
            self.reward_summary = tf.summary.scalar("reward", tf.squeeze(self.reward[0]))
            self.summary_writer = tf.summary.FileWriter("policy_logs", graph=tf.get_default_graph())

        self.train_step = tf.train.AdamOptimizer(self.policy_lr).minimize(self.loss)

        ilpo_var_list = []
        policy_var_list = []

        # Restore ILPO params and initialize policy params.
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if "ilpo" in var.name:
                ilpo_var_list.append(var)
            else:
                policy_var_list.append(var)

        saver = tf.train.Saver(var_list=ilpo_var_list)
        checkpoint = tf.train.latest_checkpoint(self.checkpoint)
        saver.restore(sess, checkpoint)
        sess.run(tf.variables_initializer(policy_var_list))

        self.deprocessed_outputs = [tf.image.convert_image_dtype(deprocess(output), dtype=tf.uint8, saturate=True) for output in self.model.outputs]
        
        if name is None:
            name = str(self.game).split('<')[-1].replace('>', '')
        self.board = Board(f'ILPO-{name}', './tmp/board/', delete=True)

    def encode(self, state):
        """Runs an encoding on a state."""

        with tf.variable_scope("ilpo_loss", reuse=True):
            with tf.variable_scope("state_encoding"):
                return self.create_encoder(state)

    def min_action(self, state, action, next_state):
        """Find the minimum action for training."""

        # Given state and action, find the closest predicted next state to the real one.
        # Use the real action as a training label for remapping the action label.
        fake_next_states = self.sess.run(self.deprocessed_outputs, feed_dict={self.inputs: [state]})

        if self.use_encoding:
            next_state_encoding = self.sess.run(self.state_encoding, feed_dict={self.inputs: [next_state]})
            fake_state_encodings = [self.sess.run(
                self.state_encoding,
                feed_dict={self.inputs: fake_next_state}) for fake_next_state in fake_next_states]
            distances = [np.linalg.norm(next_state_encoding - fake_state_encoding) for fake_state_encoding in fake_state_encodings]
        else:
            distances = [np.linalg.norm(next_state - fake_next_state) for fake_next_state in fake_next_states]
        min_action = np.argmin(distances)
        min_state = fake_next_states[min_action][0]

        if self.verbose:
            display_states = [cv2.cvtColor(cv2.resize(fake_next_state[0], (128, 128)), cv2.COLOR_RGB2BGR) for fake_next_state in fake_next_states]
            cv2.imshow("outputs", np.hstack(display_states))
            cv2.imshow("state", cv2.cvtColor(cv2.resize(state, (128, 128)), cv2.COLOR_RGB2BGR))
            cv2.imshow("NextPrediction", np.hstack([
                cv2.cvtColor(cv2.resize(next_state, (128, 128)), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(cv2.resize(min_state, (128, 128)), cv2.COLOR_RGB2BGR)]))
            cv2.waitKey(0)
        return min_action

    def action_remap_net(self, state, action, fake_action):
        """Network for remapping incorrect action labels."""

       # fake_state_encoding = tf.stop_gradient(slim.flatten(lrelu(self.encode(state)[-1], .2)))
        with tf.variable_scope("action_remap"):
            fake_state_encoding = lrelu(slim.flatten(self.create_encoder(state)[-1]), .2)
            fake_action_one_hot = tf.one_hot(fake_action, self.n_actions)
            fake_action_one_hot = lrelu(fully_connected(fake_action_one_hot, int(fake_state_encoding.shape[-1])), .2)
            real_action_one_hot = tf.one_hot(action, self.real_actions, dtype="float32")
            fake_state_action = tf.concat([fake_state_encoding, fake_action_one_hot], axis=-1)
            prediction = lrelu(fully_connected(fake_state_action, 64), .2)
            prediction = lrelu(fully_connected(prediction, 64), .2)
            prediction = fully_connected(prediction, self.real_actions)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_action_one_hot, logits=prediction))

            return tf.nn.softmax(prediction), loss

    def P(self, state):
        """Returns the next_state probabilities for a state."""

        return self.sess.run(self.model.actions, feed_dict={self.inputs: [state]})[0]

    def greedy(self, state):
        """Returns the greedy remapped action for a state."""

        p_state = self.P(state)
        action = np.argmax(p_state)

        remapped_action = self.sess.run(self.action_label, feed_dict={self.state: [state], self.fake_action: [action]})[0]

        if self.verbose:
            print(self.P(state))
            print(remapped_action)
            print("\n")

        return np.argmax(remapped_action)

    def render(self, obs):
        pass
        #self.viewer.imshow(obs)

    def eval_policy(self, game, mazes, soft=False):
        """Evaluate the policy."""

        total_reward, ratio = 0, 0
        for maze in mazes:
            terminal = False
            game.reset()
            game.load(maze)

            if soft:
                w, h = game.shape
                game.change_start_and_goal(min_distance=(w + h) // 2)

            obs = game.render('rgb_array')
            obs = np.squeeze(obs)
            steps = 0

            episode_reward = 0
            while not terminal and steps < 200:
                obs = np.squeeze(obs)
                obs = cv2.resize(obs, (128, 128))

                if np.random.uniform(0,1) <= .9:
                    action = self.greedy(obs)
                else:
                    action = game.action_space.sample()

                state, reward, terminal, _ = game.step(action)
                obs = game.render('rgb_array')

                episode_reward += reward
                steps +=1

            total_reward += episode_reward
            ratio += (state[:2] == game.end).all()

        return total_reward / len(mazes), ratio / len(mazes)

    def run_policy(self, times=10):
        """Run the policy."""
        mazes = np.repeat(self.mazes, times, axis=0)
        np.random.shuffle(mazes)

        for idx, maze in enumerate(mazes):
            terminal = False
            self.game.reset()
            self.game.load(maze)
            obs = self.game.render('rgb_array')
            obs = np.squeeze(obs)
            obs = cv2.resize(obs, (128, 128))

            total_reward = 0
            D = deque()
            episode = 0
            epsilon = .2

            prev_obs = obs.copy()

            t = 0
            while not terminal:           
                prev_obs = np.copy(obs)

                if np.random.uniform(0,1) > epsilon and t > 0:
                    action = self.greedy(obs)
                else:
                    action = self.game.action_space.sample()

                if epsilon > .2 and t >= 0:
                    epsilon -= (.2 - .2) / 1000

                obs, reward, terminal, _ = self.game.step(action)
                obs = self.game.render('rgb_array')

                obs = np.squeeze(obs)
                obs = cv2.resize(obs, (128, 128))
                total_reward += reward
                fake_action = self.min_action(prev_obs, action, obs)

                D.append((prev_obs, action, fake_action))

                if len(D) >= self.batch_size and t > 0:
                    minibatch = random.sample(D, self.batch_size)
                    obs_batch = [d[0] for d in minibatch]
                    action_batch = [d[1] for d in minibatch]
                    fake_action_batch = [d[2] for d in minibatch]

                    _, loss_summary, loss = self.sess.run([
                            self.train_step, 
                            self.loss_summary, 
                            self.loss
                        ],
                            feed_dict = {
                            self.state: obs_batch,
                            self.action: action_batch,
                            self.fake_action: fake_action_batch
                        }
                    )
                    self.board.add_scalars(
                        prior='Policy Train',
                        epoch='train',
                        loss=loss,
                    )
                    self.board.step(epoch='train')

                t += 1

            if t % 200 == 0 and t >= 0:
                aer, ratio = self.eval_policy(self.game, self.mazes)
                self.board.add_scalars(
                    prior='Policy Eval',
                    epoch='eval',
                    AER=aer,
                    ratio=ratio
                )
                aer, ratio = self.eval_policy(self.game, self.mazes, True)
                self.board.add_scalars(
                    prior='Policy Soft Generalization',
                    epoch='eval',
                    AER=aer,
                    ratio=ratio
                )
                aer, ratio = self.eval_policy(self.game, self.eval_mazes)
                self.board.add_scalars(
                    prior='Policy Hard Generalization',
                    epoch='eval',
                    AER=aer,
                    ratio=ratio
                )

                self.game.reset()
                self.game.load(maze)
                obs = self.game.render('rgb_array')
                obs = cv2.resize(obs, (128, 128))
                self.board.step(epoch='eval')


def create_dataset(path, file, output_dir):
    from PIL import Image
    from tqdm import tqdm

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = np.load(f'{path}/{file}', allow_pickle=True)
    dataset = np.repeat(dataset, 10, axis=0)
    for idx, data in enumerate(tqdm(dataset)):
        _, _, s, _, ns = data.astype(int)
        state = np.load(f'{path}/{s}.npy')
        next_state = np.load(f'{path}/{ns}.npy')
        new_state = np.hstack((state, next_state))
        Image.fromarray(new_state).save(f'{output_dir}/{idx}.png')
