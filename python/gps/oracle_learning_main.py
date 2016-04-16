import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.optimizers import Adam


def main():
    train_net_arm()
    #run_trained_policy()


class KerasPolicy:
    def __init__(self, model, goal):
        self.model = model
        self.goal = goal
        print self.goal

    def act(self, x, obs, t, noise):
        obs = np.expand_dims(obs, 0)
        return self.model.predict([self.goal, obs])[0]


def train_net_arm():
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/box2d_badmm_examplee/data_files/master_chief_model/'))
    the_data = np.load(save_path + '/the_data.npy')
    the_actions = np.load(save_path + '/the_actions.npy')
    the_goals = np.load(save_path + '/the_goals.npy')
    X_train_obs = the_data.reshape(the_data.shape[0]*the_data.shape[1], the_data.shape[2])
    X_train_goals = the_goals.reshape(the_goals.shape[0]*the_goals.shape[1], the_goals.shape[2])
    y_train = the_actions.reshape(the_actions.shape[0]*the_actions.shape[1], the_actions.shape[2])
    obs_dims = (4,)
    goal_dims = (2,)
    dim_action = 2

    goal_encoder = Sequential()
    goal_encoder.add(Dense(10, input_shape=goal_dims, activation='relu'))

    obs_encoder = Sequential()
    obs_encoder.add(Dense(30, input_shape=obs_dims, activation='relu'))
    obs_encoder.add(Dense(30, activation='relu'))

    decoder = Sequential()
    decoder.add(Merge([goal_encoder, obs_encoder], mode='concat'))
    decoder.add(Dense(32, activation='relu'))
    decoder.add(Dense(dim_action))

    decoder.compile(loss='mse', optimizer='adam')

    decoder.fit([X_train_goals, X_train_obs], y_train,
                nb_epoch=50, batch_size=64,
                show_accuracy=True, shuffle=True)
    #score = model.evaluate(X_test, y_test, batch_size=16)

    json_string = decoder.to_json()
    open(save_path + '/oracle_policy.json', 'w').write(json_string)
    decoder.save_weights(save_path + '/oracle_weights.h5')


def run_trained_policy_arm():
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/box2d_badmm_examplee/data_files/master_chief_model/'))
    from keras.models import model_from_json
    model = model_from_json(open(save_path + '/oracle_policy.json').read())
    model.load_weights(save_path + '/oracle_weights.h5')
    goals = [np.array([-3.14/2, 0.1]), np.array([0.5, 0.3]), np.array([-2.0/2, -1.0])]
    pol = KerasPolicy(model, goal=goals[0])
    agent, dO, dU = init_mujoco_agent_particle()
    cond = [0, 1, 2, 3]
    for condi in cond:
        agent.sample(pol, condi, save=False)


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


def init_mujoco_agent_particle():
    import sys
    sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
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


def train_net_pointmass():
    pass


def run_trained_policy_pointmass():
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/box2d_badmm_examplee/data_files/master_chief_model/'))
    from keras.models import model_from_json
    model = model_from_json(open(save_path + '/oracle_policy.json').read())
    model.load_weights(save_path + '/oracle_weights.h5')
    goals = [np.array([-3.14/2, 0.1]), np.array([0.5, 0.3]), np.array([-2.0/2, -1.0])]
    pol = KerasPolicy(model, goal=goals[0])
    agent, dO, dU = init_mujoco_agent_particle()
    cond = [0, 1, 2, 3]
    for condi in cond:
        agent.sample(pol, condi, save=False)


def train_net_chess():
    pass


def run_trained_policy_chess():
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/box2d_badmm_examplee/data_files/master_chief_model/'))
    from keras.models import model_from_json
    model = model_from_json(open(save_path + '/oracle_policy.json').read())
    model.load_weights(save_path + '/oracle_weights.h5')
    goals = [np.array([-3.14/2, 0.1]), np.array([0.5, 0.3]), np.array([-2.0/2, -1.0])]
    pol = KerasPolicy(model, goal=goals[0])
    agent, dO, dU = init_mujoco_agent_particle()
    cond = [0, 1, 2, 3]
    for condi in cond:
        agent.sample(pol, condi, save=False)


def init_mujoco_agent_chess():
    import sys
    sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
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
    main()


