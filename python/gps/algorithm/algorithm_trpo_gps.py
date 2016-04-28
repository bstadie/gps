import copy
import logging

import numpy as np
import scipy as sp
import tensorflow as tf

from gps.algorithm.policy.lin_guass_init import init_from_known_traj
from gps.algorithm.policy_opt.tf_model_example import trop_gps_tf_network
from gps.algorithm.policy.tf_policy import TfPolicy



class AlgorithmTRPOGPS:
    """
    Some crazy ass TRPO-GPS hybrid.
    """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.device_string = "/gpu:" + str(0)
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        self.dO = self._hyperparams['dO'] = agent.dO
        self.num_samples = self._hyperparams['samples']
        self.agent_hyper = agent
        self.lin_gauss_weights = []
        self.lin_gauss_bias = []
        self.lin_gauss_states = None
        self.lin_gauss_actions = None
        self.loss_ops = []
        self.K = None
        self.k = None
        self.init_lin_gauss_arch()
        self.sess = tf.Session()
        self.net_policy = self.init_net_pol()
        self.sess.run(tf.initialize_all_variables())

    def init_net_pol(self):
        tf_map = trop_gps_tf_network(dim_input=self.dO, dim_output=self.dU)
        return TfPolicy(self.dU, tf_map.input_tensor, tf_map.output_op, np.zeros(self.dU),
                        self.sess, self.device_string)

    def init_agent(self, agent_hyper):
        self.agent = None

    def init_lin_gauss_arch(self):
        w_shape = (self.T, self.dU, self.dX)
        self.K = np.zeros(w_shape)
        self.k = np.zeros((self.T, self.dU))
        for time_step in range(0, self.T):
            w_init = np.random.randn(self.num_samples, w_shape[1], w_shape[2])
            b_init = np.zeros(self.num_samples, w_shape[2])
            self.lin_gauss_weights.append(tf.Variable(initial_value=w_init))
            self.lin_gauss_bias.append(tf.Variable(initial_value=b_init))
            the_loss = batched_matrix_vector_multiply(self.lin_gauss_states[time_step], self.lin_gauss_weights[time_step])
            the_loss = tf.add(the_loss, self.lin_gauss_bias[time_step])
            loss = tf.pow(tf.sub(the_loss, self.lin_gauss_actions[time_step]), 2)
            loss = tf.reduce_mean(loss)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
            self.loss_ops.append(optimizer)
        self.lin_gauss_states = tf.placeholder("float", [None, self.dX])
        self.lin_gauss_actions = tf.placeholder("float", [None, self.dU])

    def iteration(self, goal_ee, x0):
        """
        Run iteration of BADMM-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        self.agent_hyper.update({'goal_ee': goal_ee})
        self.agent_hyper.update({'x0': x0})
        self.init_agent(self.agent_hyper)
        trajs = self.sample_trajectories_from_net_pol(self.agent_hyper['samples'], T=self.agent_hyper['T'])
        self.linearize_net_pol_to_lin_guass_pol(trajs)
        self.update_lin_gauss_pol()
        self.update_dynamics()
        self.sample_updated_lin_gauss_pol()
        self.train_net_on_sample()

    def sample_trajectories_from_net_pol(self, num_samples, T=100):
        trajs = []
        for iter_step in range(0, num_samples):
            trajs.append(self.agent.sample(self.net_policy, 0, save=False))
        return trajs

    def linearize_net_pol_to_lin_guass_pol(self, trajs):
        training_iters = 20
        for time_step in range(0, self.T):
            states = np.zeros(shape=(self.num_samples, self.dX))
            actions = np.zeros(shape=(self.num_samples, self.dU))
            for sample_step in range(0, len(trajs)):
                states[sample_step] = trajs[sample_step].get_X()[time_step]
                actions[sample_step] = trajs[sample_step].get_U()[time_step]
            loss_op = self.loss_ops[time_step]
            for training_step in range(0, training_iters):
                self.sess.run(loss_op, feed_dict={self.lin_gauss_states: states, self.lin_gauss_actions: actions})

            k_t, b_t = self.sess.run([self.lin_gauss_weights[time_step], self.lin_gauss_bias[time_step]])
            self.K[time_step] = k_t
            self.k[time_step] = b_t
        return init_from_known_traj(self.K, self.k, init_var=5.0, T=self.T, dU=self.dU)

    def update_lin_gauss_pol(self):
        pass

    def sample_updated_lin_gauss_pol(self):
        pass

    def train_net_on_sample(self):
        pass


def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.batch_matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result


def get_hyper_params_chess():
    pass

def testing_shit():
    agent = get_hyper_params_chess()
    hyper = {
        'T': 100,
        'dU': 7,
        'dX': 10,
        'dO': 10,
        'agent': agent
    }
    alg = AlgorithmTRPOGPS(hyper)

def main():
    hyper_params = get_hyper_params_chess()

