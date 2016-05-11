import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.optimizers import Adam
import sys

sys.path.append('/'.join(str.split(__file__, '/')[:-2]))


class KerasPolicy:
    def __init__(self, model, goal):
        self.model = model
        self.model.compile(loss='mse', optimizer='adam', batch_size=1)
        self.goal = goal
        self.goal = np.expand_dims(self.goal.flatten(), 0)

    def act(self, x, obs, t, noise):
        obs = np.expand_dims(obs, 0)
        return self.model.predict([self.goal, obs], batch_size=1)[0]


def train_net_arm():
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/mjc_chess_grip_experiment/data_files/master_chief_model/'))
    the_data = np.load(save_path + '/the_data.npy')
    the_actions = np.load(save_path + '/the_actions.npy')
    the_goals = np.load(save_path + '/the_goals.npy')
    X_train_obs = the_data.reshape(the_data.shape[0]*the_data.shape[1], the_data.shape[2])
    X_train_goals = the_goals.reshape(the_goals.shape[0]*the_goals.shape[1], the_goals.shape[2])
    y_train = the_actions.reshape(the_actions.shape[0]*the_actions.shape[1], the_actions.shape[2])
    obs_dims = (23,)
    goal_dims = (3,)
    dim_action = 7

    goal_encoder = Sequential()
    goal_encoder.add(Dense(20, input_shape=goal_dims, activation='relu'))

    obs_encoder = Sequential()
    obs_encoder.add(Dense(50, input_shape=obs_dims, activation='relu'))
    obs_encoder.add(Dense(50, activation='relu'))

    decoder = Sequential()
    decoder.add(Merge([goal_encoder, obs_encoder], mode='concat'))
    decoder.add(Dense(32, activation='relu'))
    decoder.add(Dense(dim_action))

    decoder.compile(loss='mse', optimizer='adam')

    decoder.fit([X_train_goals, X_train_obs], y_train,
                nb_epoch=100, batch_size=64,
                show_accuracy=True, shuffle=True)
    #score = model.evaluate(X_test, y_test, batch_size=16)

    json_string = decoder.to_json()
    open(save_path + '/oracle_policy.json', 'w').write(json_string)
    decoder.save_weights(save_path + '/oracle_weights.h5')


def run_trained_policy_arm():
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/mjc_chess_grip_experiment/data_files/master_chief_model/'))
    from keras.models import model_from_json
    model = model_from_json(open(save_path + '/oracle_policy.json').read())
    model.load_weights(save_path + '/oracle_weights.h5')
    #goals = [np.array([-3.14/2, 0.1]), np.array([0.5, 0.3]), np.array([-2.0/2, -1.0])]
    #the_goal = goals[2] #np.array([-0.3, 0.8]) #goals[1] #np.array([0.0, 0.0]) #goals[0]
    num_samps = 2
    goals_one = np.array([ 0.31862029,  0.65027524, -0.28322785])
    goals_two = np.array([-0.01571355,  0.532003,   -0.2956445 ])
    #final_state_one = np.zeros((num_samps*4,))
    #final_state_two = np.zeros((num_samps*4,))
    iter_step = 0
    for goal_iter in range(0, num_samps):
        the_goal = goals_one #goals_one[goal_iter]
        pol = KerasPolicy(model=model, goal=the_goal)
        agent = init_mujoco_agent_chess(the_goal)[0]
        cond = [0]
        for condi in cond:
            samp = agent.sample(pol, condi, save=False)
            #state = samp.get_X()[99, 0:2]
            #final_state_one[iter_step] = np.sum(np.abs(state - the_goal))
            #print final_state_one[iter_step]
            iter_step += 1
            print iter_step

    #save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
    #                                         'experiments/box2d_badmm_example/data_files/master_chief_model/'))
    #np.set_printoptions(suppress=True)
    #np.savetxt(save_path + '/residuals_train.txt', final_state_one, fmt='%5.8f')

    iter_step = 0
    for goal_iter in range(0, num_samps):
        the_goal = goals_two#[goal_iter]
        pol = KerasPolicy(model=model, goal=the_goal)
        agent = init_mujoco_agent_chess(the_goal)[0]
        cond = [0]
        for condi in cond:
            samp = agent.sample(pol, condi, save=False)
            #state = samp.get_X()[99, 0:2]
            #final_state_two[iter_step] = np.sum(np.abs(state - the_goal))
            #print final_state_one[iter_step]
            iter_step += 1
            print iter_step


    #np.savetxt(save_path + '/residuals_test.txt', final_state_two, fmt='%5.8f')


def init_mujoco_agent_chess(goal_ee_point):
    from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        RGB_IMAGE, RGB_IMAGE_SIZE, GOAL_EE_POINTS
    from gps.agent.mjc.agent_mjc import AgentMuJoCo


    IMAGE_WIDTH = 80
    IMAGE_HEIGHT = 64
    IMAGE_CHANNELS = 3

    #obs_include = [JOINT_ANGLES, JOINT_VELOCITIES, GOAL_EE_POINTS]
    #obs_vector_data = obs_include[:len(obs_include)-1]
    #obs_image_data = [obs_include[-1]]
    obs_include = [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, GOAL_EE_POINTS]

    SENSOR_DIMS = {
        JOINT_ANGLES: 7,
        JOINT_VELOCITIES: 7,
        END_EFFECTOR_POINTS: 3,
        END_EFFECTOR_POINT_VELOCITIES: 3,
        ACTION: 7,
        RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
        RGB_IMAGE_SIZE: 3,
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
        'pos_body_offset': [np.array([0.0, 0.12, 0])],
        'T': 50,
        'goal_ee': goal_ee_point,
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                          END_EFFECTOR_POINT_VELOCITIES],
        'obs_include': obs_include,
        'meta_include': [],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    }
    dO = 23 #+ IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS
    dU = 7
    mjc_agent = agent['type'](agent)
    return mjc_agent, dO, dU


def main():
    #train_net_arm()
    run_trained_policy_arm()
    #run_trained_policy()

if __name__ == '__main__':
    main()