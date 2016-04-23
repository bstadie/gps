import sys
import os
import numpy as np
import pickle
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.policy_opt.tf_model_example import multi_modal_network



def get_policy_for_folder(check_path):
    #tf_map_generator = example_tf_network
    #tf_map_generator = example_tf_network
    IMAGE_WIDTH = 80
    IMAGE_HEIGHT = 64
    IMAGE_CHANNELS = 3
    from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        RGB_IMAGE, RGB_IMAGE_SIZE
    SENSOR_DIMS = {
        JOINT_ANGLES: 7,
        JOINT_VELOCITIES: 7,
        END_EFFECTOR_POINTS: 6,
        END_EFFECTOR_POINT_VELOCITIES: 6,
        ACTION: 7,
        RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
        RGB_IMAGE_SIZE: 3,
    }
    config = {
        'num_filters': [5, 10],
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES],
        'obs_image_data': [RGB_IMAGE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
    }
    pol = TfPolicy.load_policy(check_path, multi_modal_network, config=config)
    return pol


def init_box2d_arm(target_state=np.array([0, 0])):
    from gps.agent.box2d.agent_box2d import AgentBox2D
    from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, ACTION
    from gps.agent.box2d.arm_world import ArmWorld

    SENSOR_DIMS = {
        JOINT_ANGLES: 2,
        JOINT_VELOCITIES: 2,
        ACTION: 2,
    }
    agent = {
        'type': AgentBox2D,
        'target_state': target_state,
        "world": ArmWorld,
        'x0': [np.array([0.5*np.pi, 0, 0, 0]),
               np.array([0.75*np.pi, 0.5*np.pi, 0, 0]),
               np.array([np.pi, -0.5*np.pi, 0, 0]),
               np.array([1.25*np.pi, 0, 0, 0]),
               ],
        'rk': 0,
        'dt': 0.05,
        'substeps': 1,
        'conditions': 4,
        'pos_body_idx': np.array([]),
        'pos_body_offset': np.array([]),
        'T': 100,
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    }
    box2d_agent = agent['type'](agent)
    return box2d_agent


def gen_data_box2d_arm(num_samples=10):
    policies_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                                 'experiments/box2d_badmm_example/data_files/policies/'))
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/box2d_badmm_example/data_files/master_chief_model/'))
    policy_folders = os.listdir(policies_path)
    dO = 4
    dU = 2
    conditions = [0, 1, 2, 3]
    goal_state_dim = 2
    t_steps = 100
    the_data = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps,  dO))
    the_actions = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps, dU))
    the_goals = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps, goal_state_dim))
    iter_count = 0
    for folder in policy_folders:
        print folder
        pol_dict_path = policies_path + '/' + folder + '/_pol'
        pol = get_policy_for_folder(pol_dict_path)
        pol_dict = pickle.load(open(pol_dict_path, "rb"))
        goal_state = pol_dict['goal_state']
        box_agent = init_box2d_arm(target_state=goal_state)
        for samples in range(0, num_samples):
            for cond in conditions:
                one_sample = box_agent.sample(pol, cond, save=False)
                obs = one_sample.get_obs()
                U = one_sample.get_U()
                the_data[iter_count] = obs
                the_actions[iter_count] = U
                the_goals[iter_count] = goal_state
                iter_count += 1
                import time
                time.sleep(0.1)
    np.save(save_path + '/the_data', the_data)
    np.save(save_path + '/the_actions', the_actions)
    np.save(save_path + '/the_goals', the_goals)
    print 'done bitch'


def gen_data_mujoco_chess(num_samples=10):
    policies_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                                 'experiments/mjc_chess_grip_experiment/data_files/policies/'))
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/mjc_chess_grip_experiment/data_files/master_chief_model/'))
    policy_folders = os.listdir(policies_path)
    mjc_agent, dO, dU = init_mujoco_agent_chess()
    goal_state = np.array([0.25, 0.7, -0.3, 0.25, 0.7, -0.2])
    conditions = [0]
    goal_state_dim = 6
    t_steps = 100
    the_data = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps,  dO))
    the_actions = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps, dU))
    the_goals = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps, goal_state_dim))
    iter_count = 0
    final_residuals = np.zeros(shape=(num_samples*len(conditions)*len(policy_folders),))
    np.set_printoptions(suppress=True)
    for folder in policy_folders:
        print folder
        pol_dict_path = policies_path + '/' + folder + '/_pol'
        pol = get_policy_for_folder(pol_dict_path)
        pol.st_idx = range(0, 14)
        pol_dict = pickle.load(open(pol_dict_path, "rb"))
        goal_state = pol_dict['goal_state']
        for samples in range(0, num_samples):
            for cond in conditions:
                one_sample = mjc_agent.sample(pol, cond, save=False)
                obs = one_sample.get_obs()
                U = one_sample.get_U()
                the_data[iter_count] = obs
                the_actions[iter_count] = U
                the_goals[iter_count] = goal_state
                residual = np.sum(np.abs(goal_state - one_sample.get_X()[99, 14:20]))
                final_residuals[iter_count] = residual
                print residual
                iter_count += 1
                import time
                time.sleep(0.1)
    np.save(save_path + '/the_data', the_data)
    np.save(save_path + '/the_actions', the_actions)
    np.save(save_path + '/the_goals', the_goals)
    np.savetxt(save_path + '/residuals_train.txt', final_residuals, fmt='%5.8f')
    print 'done bitch'


def init_mujoco_agent_chess():
    from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        RGB_IMAGE, RGB_IMAGE_SIZE
    from gps.agent.mjc.agent_mjc import AgentMuJoCo


    IMAGE_WIDTH = 80
    IMAGE_HEIGHT = 64
    IMAGE_CHANNELS = 3


    SENSOR_DIMS = {
        JOINT_ANGLES: 7,
        JOINT_VELOCITIES: 7,
        END_EFFECTOR_POINTS: 6,
        END_EFFECTOR_POINT_VELOCITIES: 6,
        ACTION: 7,
        RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
        RGB_IMAGE_SIZE: 3,
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
        'pos_body_offset': [np.array([0, 0.2, 0])],
        'T': 100,
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                          END_EFFECTOR_POINT_VELOCITIES],
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE],
        'meta_include': [RGB_IMAGE_SIZE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    }
    dO = 14 + IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS
    dU = 7
    mjc_agent = agent['type'](agent)
    return mjc_agent, dO, dU


def gen_data_mujoco_pointmass(num_samples=100):
    policy_folders = os.listdir(policies_path)
    mjc_agent, dO, dU = init_mujoco_agent_pointmass()
    conditions = [0, 1, 2, 3]
    goal_state_dim = 2
    t_steps = 100
    the_data = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps,  dO))
    the_actions = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps, dU))
    the_goals = np.zeros((num_samples*len(conditions)*len(policy_folders), t_steps, goal_state_dim))
    iter_count = 0
    for folder in policy_folders:
        print folder
        pol_dict_path = policies_path + '/' + folder + '/_pol'
        pol = get_policy_for_folder(pol_dict_path)
        pol_dict = pickle.load(open(pol_dict_path, "rb"))
        goal_state = pol_dict['goal_state']
        for samples in range(0, num_samples):
            for cond in conditions:
                one_sample = mjc_agent.sample(pol, cond, save=False)
                obs = one_sample.get_obs()
                U = one_sample.get_U()
                the_data[iter_count] = obs
                the_actions[iter_count] = U
                the_goals[iter_count] = goal_state
                iter_count += 1
                import time
                time.sleep(0.5)
    np.save(save_path + '/the_data', the_data)
    np.save(save_path + '/the_actions', the_actions)
    np.save(save_path + '/the_goals', the_goals)
    print 'done bitch'


def init_mujoco_agent_pointmass():
    from gps.agent.mjc.agent_mjc import AgentMuJoCo
    from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, ACTION
    SENSOR_DIMS = {
        JOINT_ANGLES: 2,
        JOINT_VELOCITIES: 2,
        ACTION: 2,
    }
    agent = {
        'type': AgentMuJoCo,
        'filename': './mjc_models/particle2d.xml',
        'x0': [np.array([0., 0., 0., 0.]), np.array([0., 1., 0., 0.]),
               np.array([1., 0., 0., 0.]), np.array([1., 1., 0., 0.])],
        'dt': 0.05,
        'substeps': 5,
        'conditions': 4,
        'T': 100,
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    }
    dO = 4
    dU = 2
    mjc_agent = agent['type'](agent)
    return mjc_agent, dO, dU


if __name__ == '__main__':
    gen_data_mujoco_chess(num_samples=16)
