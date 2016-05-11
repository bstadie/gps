
import mjcpy
world = mjcpy.MJCWorld('/home/bradly/gps/mjc_models/walker.xml')
model = world.get_model()
print model['nq']
print world.get_data()['site_xpos']
#print model.data.qpos.ravel()
#print model['qpos']
print world.get_data()['qpos'].ravel()
print world.get_data()['qvel'].ravel().shape