import copy
import logging

import numpy as np
import scipy as sp
import tensorflow as tf

import sys

sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.algorithm.policy_opt.tf_model_example import trpo_gps_tf_network
from gps.algorithm.policy.lin_gauss_init import init_from_known_traj
from gps.algorithm.policy.tf_policy import TfPolicy


class AlgorithmTRPOGPS:
    """
    Some crazy ass TRPO-GPS hybrid.
    """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.agent = None
        self.device_string = "/gpu:" + str(0)
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T']
        self.dU = self._hyperparams['dU']
        self.dX = self._hyperparams['dX']
        self.dO = self._hyperparams['dO']
        self.num_samples = self._hyperparams['samples']
        self.agent_hyper = agent
        self.lin_gauss_weights = []
        self.lin_gauss_bias = []
        self.lin_gauss_states = []
        self.lin_gauss_actions = []
        self.lin_gauss_pol = None
        self.loss_ops = []
        self.loss_tensors = []
        self.K = None
        self.k = None
        self.init_lin_gauss_arch()
        self.init_dynamics()
        self.sess = tf.Session()
        self.net_policy = self.init_net_pol()
        self.sess.run(tf.initialize_all_variables())

    def init_net_pol(self):
        tf_map = trpo_gps_tf_network(dim_input=self.dO, dim_output=self.dU)
        return TfPolicy(self.dU, tf_map.input_tensor, tf_map.output_op, np.zeros(self.dU),
                        self.sess, self.device_string)

    def init_agent(self, agent_hyper):
        self.agent = agent_hyper['type'](agent_hyper)

    def init_lin_gauss_arch(self):
        w_shape = (self.T, self.dU, self.dX)
        self.K = np.zeros(w_shape)
        self.k = np.zeros((self.T, self.dU))
        for time_step in range(0, self.T):
            w_init_shape = (1, w_shape[1], w_shape[2])
            b_init_shape = (1, w_shape[1])
            w = tf.Variable(tf.random_normal(w_init_shape, stddev=0.01))
            b = tf.Variable(tf.zeros(b_init_shape, dtype='float'))
            w = tf.tile(w, [self.num_samples, 1, 1])
            b = tf.tile(b, [self.num_samples, 1])
            self.lin_gauss_weights.append(w)
            self.lin_gauss_bias.append(b)
            self.lin_gauss_states.append(tf.placeholder("float", [None, self.dX]))
            self.lin_gauss_actions.append(tf.placeholder("float", [None, self.dU]))
            the_loss = batched_matrix_vector_multiply(vector=self.lin_gauss_states[time_step],
                                                      matrix=self.lin_gauss_weights[time_step])
            the_loss = tf.add(the_loss, self.lin_gauss_bias[time_step])
            loss = tf.pow(tf.sub(the_loss, self.lin_gauss_actions[time_step]), 2)
            loss = tf.cast(loss, 'float32')
            loss = tf.reduce_mean(loss)
            optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.4).minimize(loss)
            #tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
            self.loss_ops.append(optimizer)
            self.loss_tensors.append(loss)

    def init_dynamics(self):
        pass

    def iteration(self, goal_ee, x0):
        """
        Run iteration of BADMM-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        self.agent_hyper.update({'goal_ee': goal_ee})
        self.agent_hyper.update({'x0': x0})
        self.init_agent(self.agent_hyper)
        trajs = self.sample_trajectories_from_net_pol(self.agent_hyper['samples'])
        self.lin_gauss_pol = self.linearize_net_pol_to_lin_guass_pol(trajs)
        self.fit_dynamics()
        self.update_lin_gauss_pol()
        self.update_dynamics_prior()
        self.sample_updated_lin_gauss_pol()
        self.train_net_on_sample()

    def sample_trajectories_from_net_pol(self, num_samples):
        trajs = []
        for iter_step in range(0, num_samples):
            trajs.append(self.agent.sample(self.net_policy, 0, save=False, verbose=True))
        return trajs

    def linearize_net_pol_to_lin_guass_pol(self, trajs):
        training_iters = 10
        for time_step in range(0, self.T):
            states = np.zeros(shape=(self.num_samples, self.dX))
            actions = np.zeros(shape=(self.num_samples, self.dU))
            for sample_step in range(0, len(trajs)):
                states[sample_step] = trajs[sample_step].get_X()[time_step]
                actions[sample_step] = trajs[sample_step].get_U()[time_step]
            loss_op = self.loss_ops[time_step]
            actual_loss = self.loss_tensors[time_step]
            for training_step in range(0, training_iters):
                al = self.sess.run([loss_op, actual_loss], feed_dict={self.lin_gauss_states[time_step]: states,
                                                                      self.lin_gauss_actions[time_step]: actions})[1]
            print al
            k_t, b_t = self.sess.run([self.lin_gauss_weights[time_step], self.lin_gauss_bias[time_step]])
            self.K[time_step] = k_t[0]
            self.k[time_step] = b_t[0]
        print np.var(actions)/(100*2*7*5000)
        return init_from_known_traj(self.K, self.k, init_var=np.var(actions), T=self.T, dU=self.dU)

    def update_dynamics_prior(self):
        pass

    def fit_dynamics(self):
        pass

    def update_lin_gauss_pol(self):
        pass

    def sample_updated_lin_gauss_pol(self, num_samples=None):
        trajs = []
        if num_samples is None:
            num_samples = self.num_samples
        for iter_step in range(0, num_samples):
            trajs.append(self.agent.sample(self.lin_gauss_pol, 0, save=False))
        return trajs

    def train_net_on_sample(self):
        pass


def batched_matrix_vector_multiply(vector, matrix):
    """ computes Ax in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [2])
    mult_result = tf.batch_matmul(matrix, vector_batch_as_matricies)
    squeezed_result = tf.squeeze(mult_result, [2])
    return squeezed_result


def get_hyper_params_chess():
    from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        RGB_IMAGE, RGB_IMAGE_SIZE, GOAL_EE_POINTS
    from gps.agent.mjc.agent_mjc import AgentMuJoCo

    obs_include = [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, GOAL_EE_POINTS]

    SENSOR_DIMS = {
        JOINT_ANGLES: 7,
        JOINT_VELOCITIES: 7,
        END_EFFECTOR_POINTS: 6,
        END_EFFECTOR_POINT_VELOCITIES: 6,
        ACTION: 7,
        GOAL_EE_POINTS: 3,
    }

    agent = {
        'type': AgentMuJoCo,
        'filename': './mjc_models/pr2_gripping.xml',
        'x0': np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                              np.zeros(7)]),
        'dt': 0.05,
        'substeps': 5,
        'conditions': 1,
        'train_conditions': [0],
        'test_conditions': [0],
        'pos_body_idx': np.array([1]),
        'pos_body_offset': [np.array([-0.13, -0.08, 0])],
        'T': 100,
        'goal_ee': np.array([0, 0, 1]),
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                          END_EFFECTOR_POINT_VELOCITIES],
        'obs_include': obs_include,
        'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    }
    return agent


def load_pol():
    import os
    policies_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                                 'experiments/mjc_chess_grip_experiment/data_files/policies/9/'))

    pol_dict_path = policies_path + '/_pol'
    return TfPolicy.load_policy(pol_dict_path, trpo_gps_tf_network)


def testing_shit():
    agent = get_hyper_params_chess()
    hyper = {
        'T': 100,
        'dU': 7,
        'dX': 26,
        'dO': 29,
        'samples': 5,
        'agent': agent
    }
    alg = AlgorithmTRPOGPS(hyper)
    alg.init_agent(agent)
    alg.net_policy = load_pol()
    trajs = alg.sample_trajectories_from_net_pol(hyper['samples'])
    alg.lin_gauss_pol = alg.linearize_net_pol_to_lin_guass_pol(trajs)
    alg.fit_dynamics(trajs)
    alg.update_lin_gauss_pol()
    trajs2 = alg.sample_updated_lin_gauss_pol(hyper['samples'])
    alg.update_dynamics_prior([trajs, trajs2])



testing_shit()

