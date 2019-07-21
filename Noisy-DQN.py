
import numpy as np
import gym
import tensorflow as tf
import random
from collections import deque

class DQN():
    def __init__(self, gym_game, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, discount_factor=0.9, num_of_episodes=500):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.num_of_episodes = num_of_episodes
        self.game = gym_game
        self.environment = gym.make(gym_game)
        try:
            try:
                shape = self.environment.observation_space.shape
                self.state_size = (shape[0], shape[1], shape[2])
                self.s = shape[0]*shape[1]*shape[2]
                self.state_mode = "observation"
            except:
                self.state_size = self.environment.observation_space.shape[0]
                self.s = self.state_size
                self.state_mode = "information"
            self.state_container = "box"
        except:
            self.state_size = self.environment.observation_space.n
            self.state_mode = "information"
            self.state_container = "discrete"

        try:
            self.action_size = self.environment.action_space.shape[0]
        except:
            try:
                self.action_size = self.environment.action_space.n
            except:
                self.action_size = self.environment.action_space.shape[0]
        self.a = self.action_size

        print("state size is: ",self.state_size)
        print("action size is: ", self.action_size)
        self.memory = deque(maxlen=20000)

        self.create_model()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    def create_model(self):
        """
        neural network model
        :return:
        """
        try:
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size[0], self.state_size[1], self.state_size[2]])
            self.target = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size])
        except:
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.target = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size])

        if self.state_mode == "information":
            # if the state is not an image)
            out = tf.nn.relu(self.noisy_dense(24, self.input))
            out = tf.nn.relu(self.noisy_dense(24, out))
            self.out = self.noisy_dense(self.action_size, out)
        elif self.state_mode == "observation":
            # if the state is an image
            out = tf.layers.conv2d(inputs=self.input, filters=128, kernel_size=(5,5), activation="relu")
            out = tf.layers.max_pooling2d(inputs=out,pool_size=(2,2),strides=(2,2))
            out = tf.layers.conv2d(inputs=self.input, filters=64, kernel_size=(3,3), activation="relu")
            out = tf.layers.max_pooling2d(inputs=out,pool_size=(2,2),strides=(2,2))
            out = tf.layers.flatten(inputs=out)
            out = tf.nn.relu(self.noisy_dense(64, out))
            out = tf.nn.relu(self.noisy_dense(64, out))
            self.out = self.noisy_dense(self.action_size, out)

        loss = tf.reduce_mean(tf.square(tf.subtract(self.out, self.target)))
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)

    def noisy_dense(self, units, input):
        w_shape = [units, input.shape[1].value]
        mu_w = tf.Variable(initial_value=tf.truncated_normal(shape=w_shape))
        sigma_w = tf.Variable(initial_value=tf.constant(0.017, shape=w_shape))
        epsilon_w = tf.random_uniform(shape=w_shape)

        b_shape = [units]
        mu_b = tf.Variable(initial_value=tf.truncated_normal(shape=b_shape))
        sigma_b = tf.Variable(initial_value=tf.constant(0.017, shape=b_shape))
        epsilon_b = tf.random_uniform(shape=b_shape)

        w = tf.add(mu_w, tf.multiply(sigma_w, epsilon_w))
        b = tf.add(mu_b, tf.multiply(sigma_b, epsilon_b))

        return tf.matmul(input, tf.transpose(w)) + b

    def fit(self, input, target):
        self.sess.run(self.optimizer, feed_dict={self.input: input, self.target: target})

    def predict(self, input):
        return self.sess.run(self.out, feed_dict={self.input: input})

    def state_reshape(self, state):
        shape = state.shape
        if self.state_mode == "observation":
            return np.reshape(state, [1, shape[0], shape[1], shape[2]])
        elif self.state_mode == "information":
            return np.reshape(state, [1, shape[0]])

    def act(self, state):
        """
        act randomly by probability of epsilon or predict the next move by the neural network model
        :param state:
        :return:
        """
        return np.argmax(self.predict(state)[0])

    def remember(self, state, next_state, action, reward, done):
        """
        remember the experience
        :param state:
        :param next_state:
        :param action:
        :param reward:
        :param done:
        :return:
        """
        self.memory.append((state, next_state, action, reward, done))

    def replay(self):
        """
        experience replay. find the q-value and train the neural network model with state as input and q-values as targets
        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        batch = random.choices(self.memory,k=self.batch_size)
        for state, next_state, action, reward, done in batch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.max(self.predict(next_state)[0])
            final_target = self.predict(state)
            final_target[0][action] = target
            self.fit(state,final_target)

    def play(self):
        """
        play for num_of_episodes. remember the experiences in each episode. replay the experience at the end of the episode
        :return:
        """
        for episode in range(self.num_of_episodes+1):
            state = self.environment.reset()
            state = self.state_reshape(state)
            r = []
            t = 0
            while True:
                action = self.act(state)
                next_state, reward, done, _ = self.environment.step(action)
                next_state = self.state_reshape(next_state)
                self.remember(state, next_state, action, reward, done)
                state = next_state
                r.append(reward)
                t += 1
                if done:
                    r = np.mean(r)
                    print("episode number: ", episode,", reward: ",r , "time score: ", t)
                    self.save_info(episode, r, t)
                    break
            self.replay()

    def save_info(self, episode, reward, time):
        file = open("./Plot/NoisyDQN-"+self.game+"-"+str(self.num_of_episodes)+"-episodes-batchsize-"+str(self.batch_size), 'a')
        file.write(str(episode)+" "+str(reward)+" "+str(time)+" \n")
        file.close()


game = "Pong-v0" # bd
# game = "CartPole-v1" # bd
dqn = DQN(game, num_of_episodes=5000)
dqn.play()
