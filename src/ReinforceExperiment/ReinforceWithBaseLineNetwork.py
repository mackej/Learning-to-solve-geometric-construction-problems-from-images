import numpy as np
import tensorflow as tf


class Network:
    def __init__(self, env, args):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            with self.session.as_default():
                self.image_and_masks_input = tf.keras.layers.Input(shape=env.image_and_masks_shape,
                                                                   name='image_and_masks_input')

                self.class_ids_input = tf.keras.layers.Input(shape=env.class_ids_shape, name='class_ids_input')
                self.tool_mask_input = tf.keras.layers.Input(shape=env.tool_mask_shape, name='tool_mask_input')
                self.conv_head = self.buil_conv_head(env, args)

                self.build_model(env, args)

                self.build_baseline(env, args)

    def buil_conv_head(self, env, args):

        conv = tf.keras.layers.Conv2D(20, (3, 3), strides=1, padding="same")(self.image_and_masks_input)
        conv = tf.keras.layers.Activation("relu")(conv)
        conv = tf.keras.layers.Conv2D(20, (3, 3), strides=1, padding="same")(conv)
        conv = tf.keras.layers.Activation("relu")(conv)
        conv = tf.keras.layers.Conv2D(30, (3, 3), strides=2, padding="same")(conv)
        conv = tf.keras.layers.Activation("relu")(conv)

        conv = tf.keras.layers.Conv2D(40, (3, 3), strides=2, padding="same")(conv)
        conv = tf.keras.layers.Activation("relu")(conv)
        conv = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(conv)
        conv = tf.keras.layers.Flatten()(conv)

        return conv

    def build_model(self, env, args):

        concat = tf.keras.layers.Concatenate()([self.conv_head, self.class_ids_input, self.tool_mask_input])
        # hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", name="first_dense")(concat)
        # hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", name="second_dense")(hidden)

        output_1 = tf.keras.layers.Dense(env.max_number_of_actions, activation="softmax", name="click_n_one")(concat)
        output_2 = tf.keras.layers.Dense(env.max_number_of_actions, activation="softmax", name="click_n_two")(concat)
        output_3 = tf.keras.layers.Dense(env.max_number_of_actions, activation="softmax", name="click_n_three")(concat)

        output_tool = tf.keras.layers.Dense(env.total_number_of_tools, activation="softmax", name="tool_to_use")(concat)

        self.model = tf.keras.Model(inputs=[self.image_and_masks_input, self.class_ids_input, self.tool_mask_input],
                                    outputs=[output_tool, output_1, output_2, output_3])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )

    def build_baseline(self, env, args):

        concat = tf.keras.layers.Concatenate()([self.conv_head, self.class_ids_input, self.tool_mask_input])
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(concat)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(hidden)
        output = tf.keras.layers.Dense(1, activation=None)(hidden)
        self.baseline = tf.keras.Model(inputs=[self.image_and_masks_input, self.class_ids_input,
                                               self.tool_mask_input], outputs=output)

        self.baseline.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss='mse',
            experimental_run_tf_function=False
        )

    def train(self, states, actions, returns):
        returns = np.array(returns)
        states = [
                    np.array(states["image_and_mask"]),
                    np.array(states["class_ids"]),
                    np.array(states["tool_mask"]),
                ]
        actions = [
            np.array(actions["tool"]),
            np.array(actions["click_one"]),
            np.array(actions["click_two"]),
            np.array(actions["click_three"])
        ]
        with self.graph.as_default():
            with self.session.as_default():
                baseline = self.baseline.predict_on_batch(states)
                sample_w = (returns - baseline.flatten())
                self.model.train_on_batch(states, actions, sample_weight=[sample_w, sample_w, sample_w, sample_w])
                self.baseline.train_on_batch(states, returns)

    def predict(self, states):
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.predict_on_batch([
                    np.array(states["image_and_mask"]),
                    np.array(states["class_ids"]),
                    np.array(states["tool_mask"]),
                ])


