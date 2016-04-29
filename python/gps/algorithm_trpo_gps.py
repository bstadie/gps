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
from gps.algorithm.policy_opt.tf_utils import TfSolver
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.sample.sample_list import SampleList


class AlgorithmTRPOGPS:
    """
    Some crazy ass TRPO-GPS hybrid.
    """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.agent = None
        self.device_string = "/gpu:" + str(0)
        self.agent_hyper = self._hyperparams['agent_hyper']
        self.T = self._hyperparams['T']
        self.dU = self._hyperparams['dU']
        self.dX = self._hyperparams['dX']
        self.dO = self._hyperparams['dO']
        self.num_samples = self._hyperparams['samples']
        self.init_agent(agent_hyper=self.agent_hyper)

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
        self.traj_opt_alg = None

        self.solver = None
        self.tf_map = None
        self.sess = tf.Session()
        self.net_policy = self.init_net_pol()
        self.sess.run(tf.initialize_all_variables())

        #self.take_first_iter(self.agent_hyper['goal_ee'], self.agent_hyper['x0'])

    def init_net_pol(self):
        tf_map = trpo_gps_tf_network(dim_input=self.dO, dim_output=self.dU)
        self.tf_map = tf_map
        self.solver = TfSolver(loss_scalar=tf_map.get_loss_op(),
                               solver_name='adam',
                               base_lr=0.01,
                               lr_policy='fixed',
                               momentum=0.9,
                               weight_decay=0.005)
        return TfPolicy(self.dU, tf_map.input_tensor, tf_map.output_op, np.zeros(self.dU),
                        self.sess, self.device_string)

    def init_agent(self, agent_hyper):
        self.agent = agent_hyper['type'](agent_hyper)
        self._hyperparams.update({'agent': self.agent})

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

    def init_traj_opt_alg(self):
        self._hyperparams['alg_hyper'].update({'agent': self.agent})
        traj_opt = AlgorithmTrajOpt(self._hyperparams['alg_hyper'])
        if self.traj_opt_alg is not None:
            traj_opt.dynamics.prior = self.traj_opt_alg.dynamics.get_prior()
        self.traj_opt_alg = traj_opt
        if self.lin_gauss_pol is not None:
            self.traj_opt_alg.cur[0].traj_distr = self.lin_gauss_pol
        else:
            self.lin_gauss_pol = self.traj_opt_alg.cur[0].traj_distr

    def take_first_iter(self, goal_ee, x0):  # the first iteration goes backwards for stability reasons.
        self.agent_hyper.update({'goal_ee': goal_ee})
        self.agent_hyper.update({'x0': x0})
        self.init_agent(self.agent_hyper)
        self.init_traj_opt_alg()
        lin_gauss_trajs = self.sample_lin_gauss_pol(self.agent_hyper['samples'])
        self.update_lin_gauss_pol(lin_gauss_trajs)
        nu_lin_gauss_trajs = self.sample_lin_gauss_pol()
        self.train_net_on_sample(nu_lin_gauss_trajs)

    def iteration(self, goal_ee, x0):
        """
        Run iteration of policy gradient-GPS hybrid algorithm.

        Args:
            sample_lists: goal position, start position.
        """
        self.agent_hyper.update({'goal_ee': goal_ee})
        self.agent_hyper.update({'x0': x0})
        self.init_agent(self.agent_hyper)
        trajs = self.sample_trajectories_from_net_pol(self.agent_hyper['samples'])
        self.lin_gauss_pol = self.linearize_net_pol_to_lin_guass_pol(trajs)
        self.init_traj_opt_alg()
        self.fit_dynamics(trajs)
        lin_gauss_trajs = self.sample_lin_gauss_pol(self.agent_hyper['samples'])
        self.update_lin_gauss_pol(lin_gauss_trajs)
        nu_lin_gauss_trajs = self.sample_lin_gauss_pol()
        self.train_net_on_sample(nu_lin_gauss_trajs)

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
        lin_gauss = init_from_known_traj(self.K, self.k, init_var=np.var(actions), T=self.T, dU=self.dU)
        if self.lin_gauss_pol is not None:  # i have no idea how to set this shit.
            lin_gauss.pol_covar = self.lin_gauss_pol.pol_covar
            lin_gauss.chol_pol_covar = self.lin_gauss_pol.chol_pol_covar
            lin_gauss.inv_pol_covar = self.lin_gauss_pol.inv_pol_covar
        return lin_gauss

    def fit_dynamics(self, sample_list):
        t = SampleList(sample_list)
        self.traj_opt_alg.update_only_dynamics(t)

    def update_lin_gauss_pol(self, sample_list):
        t = SampleList(sample_list)
        self.traj_opt_alg.iteration(t, cheat=True)
        self.lin_gauss_pol = self.traj_opt_alg.cur[0].traj_distr

    def sample_lin_gauss_pol(self, num_samples=None):
        trajs = []
        if num_samples is None:
            num_samples = self.num_samples
        for iter_step in range(0, num_samples):
            trajs.append(self.agent.sample(self.lin_gauss_pol, 0, save=False))
        return trajs

    def train_net_on_sample(self, trajs):
        num_samples = len(trajs)
        obs = np.zeros(shape=(num_samples, self.T, self.dO))
        actions = np.zeros(shape=(num_samples, self.T, self.dU))
        for sample_step in range(0, num_samples):
            obs[sample_step] = trajs[sample_step].get_obs()
            actions[sample_step] = trajs[sample_step].get_U()
        
        obs = np.reshape(obs, (num_samples*self.T, self.dO))
        actions = np.reshape(actions, (num_samples*self.T, self.dU))
        
        batch_size = 25
        batches_per_epoch = np.floor(num_samples*self.T / batch_size)
        idx = range(num_samples*self.T)
        np.random.shuffle(idx)

        average_loss = 0
        for i in range(1000):
            # Load in data for this batch.
            start_idx = int(i * batch_size %
                            (batches_per_epoch * batch_size))
            idx_i = idx[start_idx:start_idx+batch_size]
            feed_dict = {self.tf_map.get_input_tensor(): obs[idx_i], 
                         self.tf_map.get_target_output_tensor(): actions[idx_i]}
            average_loss += self.solver(feed_dict, self.sess)
            if i % 500 == 0 and i != 0:
                print 'tf loss is ' + str(average_loss/500)
                average_loss = 0


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


def get_traj_opt_hyper(agent, T, dU, dX, dO, samples):
    from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
    from gps.algorithm.cost.cost_fk import CostFK
    from gps.algorithm.cost.cost_action import CostAction
    from gps.algorithm.cost.cost_sum import CostSum
    from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
    from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
    from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
    from gps.algorithm.policy.lin_gauss_init import init_lqr


    algorithm = {
        'type': AlgorithmTrajOpt,
        'conditions': 1,
        'iterations': 10,
        'T': T,
        'dU': dU,
        'dX': dX,
        'dO': dO,
        'samples': samples
    }

    PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])


    algorithm['init_traj_distr'] = {
        'type': init_lqr,
        'init_gains':  1.0 / PR2_GAINS,
        'init_acc': np.zeros(7),
        'init_var': 1.0,
        'stiffness': 1.0,
        'stiffness_vel': 0.5,
        'dt': agent['dt'],
        'T': agent['T'],
    }

    torque_cost = {
        'type': CostAction,
        'wu': 5e-5 / PR2_GAINS,
    }

    fk_cost = {
        'type': CostFK,
        'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
        'wp': np.array([1, 1, 1, 1, 1, 1]),
        'l1': 0.1,
        'l2': 10.0,
        'alpha': 1e-5,
    }

    algorithm['cost'] = {
        'type': CostSum,
        'costs': [torque_cost, fk_cost],
        'weights': [1.0, 1.0],
    }

    algorithm['dynamics'] = {
        'type': DynamicsLRPrior,
        'regularization': 1e-6,
        'prior': {
            'type': DynamicsPriorGMM,
            'max_clusters': 20,
            'min_samples_per_cluster': 40,
            'max_samples': 20,
        },
    }

    algorithm['traj_opt'] = {
        'type': TrajOptLQRPython,
    }

    algorithm['policy_opt'] = {}

    return algorithm


def load_pol():
    import os
    policies_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                                 'experiments/mjc_chess_grip_experiment/data_files/policies/9/'))

    pol_dict_path = policies_path + '/_pol'
    return TfPolicy.load_policy(pol_dict_path, trpo_gps_tf_network)


def testing_shit():
    agent_hyper = get_hyper_params_chess()
    T = 100
    dU = 7
    dX = 26
    dO = 29
    samples = 5
    alg_hyper = get_traj_opt_hyper(agent_hyper, T, dU, dX, dO, samples)
    hyper = {
        'T': T,
        'dU': dU,
        'dX': dX,
        'dO': dO,
        'samples': samples,
        'agent_hyper': agent_hyper,
        'alg_hyper': alg_hyper
    }
    alg = AlgorithmTRPOGPS(hyper)
    alg.take_first_iter(np.array([0, 0, 1]), np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                                                             np.zeros(7)]))
    for iter_step in range(0, 10):
        alg.iteration(np.array([0, 0, 1]), np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                                                           np.zeros(7)]))


def test_traj_opt():
    agent_hyper = get_hyper_params_chess()
    T = 100
    dU = 7
    dX = 26
    dO = 29
    samples = 5
    alg_hyper = get_traj_opt_hyper(agent_hyper, T, dU, dX, dO, samples)
    hyper = {
        'T': T,
        'dU': dU,
        'dX': dX,
        'dO': dO,
        'samples': samples,
        'agent_hyper': agent_hyper,
        'alg_hyper': alg_hyper
    }
    alg = AlgorithmTRPOGPS(hyper)
    alg.init_traj_opt_alg()
    for iter_step in range(0, 10):
        lin_gauss_trajs = alg.sample_lin_gauss_pol(hyper['samples'])
        alg.update_lin_gauss_pol(lin_gauss_trajs)


test_traj_opt()

