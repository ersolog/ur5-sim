import mujoco
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from utils.mujoco_parser import *
from utils.transformation import *
from utils.slider import *
from utils.utility import *
from utils.rrt import *
from xml.etree import ElementTree as ET
from xml.dom import minidom
np.set_printoptions(precision=2,suppress=True,linewidth=100)
plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)
print ("MuJoCo:[%s]"%(mujoco.__version__))

xml_path = 'tabletop_manipulation/scene_ur5e_cylinders.xml'
env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path,verbose=True)
print ("Done.")

# Reset
np.random.seed(seed=0)
env.reset()

# Initialize UR5e
joint_names = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
               'wrist_1_joint','wrist_2_joint','wrist_3_joint']
q0 = np.deg2rad([-7.21,-129.4,118.6,14.38,79.06,-3.59])
p0 = env.get_p_body(body_name='ur_base')+np.array([0.5,0.0,0.05])
R0 = rpy_deg2r([-180,0,90])

env.forward(q=q0,joint_names=joint_names) # update UR5e pose

# Set cylinder object poses
obj_names = env.get_body_names(prefix='obj_')
n_obj = len(obj_names)

# red cup experiment comparison
# obj_xyzs = np.array([
# [ 0.75, -0.32,  0.81 ],
#  [ 0.75, -0.26,  1.2 ]])

# easy for ur5 ex 1.
# obj_xyzs = np.array([
# [ 0.8, -0.05,  0.81 ]])

# easy for ur5 ex 2.
obj_xyzs = np.array([
[ 0.8, -0.05,  0.81 ],
 [ 0.8, -0.05,  1.2 ]])

# # weighted cup experiment comparison
# obj_xyzs = np.array([
# [ 0.7, 0.20,  0.81 ]])

for obj_idx in range(n_obj):
    pos = obj_xyzs[obj_idx,:]
    env.set_p_base_body(body_name=obj_names[obj_idx],p=obj_xyzs[obj_idx,:])
    env.set_R_base_body(body_name=obj_names[obj_idx],R=np.eye(3,3))
ob_colors = get_colors(n_obj)
env.forward() # update object poses
obj_xyzs = env.get_xyzs()
env.init_viewer(title='Double click which object to pick',
                transparent=False,distance=3.0)
while env.is_viewer_alive():
    # Update
    env.step(
        ctrl        = np.append(q0,2.0),
        joint_names = joint_names+['gripper_finger2_joint'],
    )
    xyz_click,flag_click = env.get_xyz_left_double_click()
    if flag_click: # if double clicked, get the closest body
        body_name_clicked,p_body_clicked = env.get_body_name_closest(
            xyz_click,body_names=obj_names)
    # Render
    if env.loop_every(tick_every=10):
        env.plot_time()
        if xyz_click is not None: 
            env.plot_sphere(p=xyz_click,r=0.01,rgba=(1,0,0,0.5))
            env.plot_body_T(body_name=body_name_clicked,axis_len=0.25)
        # env.plot_contact_info(r_arrow=0.005,h_arrow=0.1,plot_sphere=False)
        env.render()
# Save state
if xyz_click is not None:
    print ("[%s] has been selected to pick."%(body_name_clicked))
    print (" object base position is %s."%(p_body_clicked))
    state = env.get_state() # get env state
else:
    print ("No object has been selected.")
print ("Done.")

# Restore state and solve IKs
env.set_state(**state,step=True)
q_grasp,_,_ = solve_ik(
    env                = env,
    joint_names_for_ik = joint_names,
    body_name_trgt     = 'ur_tcp_link',
    q_init             = q0,
    p_trgt             = p_body_clicked + np.array([0,0,0.09]),
    R_trgt             = R0,
)
# Interpolate and smooth joint positions
times,traj_interp,traj_smt,times_anchor = interpolate_and_smooth_nd(
    anchors   = np.vstack([q0,q_grasp]),
    HZ        = env.HZ,
    vel_limit = np.deg2rad(30),
)
L = len(times)

# Open the gripper
qpos = env.get_qpos_joints(joint_names=joint_names)
for tick in range(100):
    env.step( # dynamic update
        ctrl        = np.append(qpos,2.0), # 0.0: close, 2.0: full open
        joint_names = joint_names+['gripper_finger2_joint'])

# Set collision configurations 
robot_body_names = env.get_body_names(prefix='ur_')
obj_body_names   = env.get_body_names(prefix='obj_')
env_body_names   = ['front_object_table','side_object_table', 'obstacle_one_fb', 'obstacle_two_fb', 'wall']
# (optional) exclude 'body_name_clicked' from 'obj_body_names'
obj_body_names.remove(body_name_clicked)

############################## IK #################################################
env.init_viewer(
    title='Checking collision while moving to the grasping pose',
    transparent=False,distance=3.0)
tick = 0
while env.is_viewer_alive():
    # Update
    time = times[tick]
    qpos = traj_smt[tick,:]
    env.forward(q=qpos,joint_names=joint_names)

    # Check collsision 
    is_feasible, __ = is_qpos_feasible(
        env,qpos,joint_names,
        robot_body_names,obj_body_names,env_body_names)
    # Render
    if (tick%5)==0 or tick==(L-1):
        env.plot_text(p=np.array([0,0,1]),
                      label='[%d/%d] time:[%.2f]sec'%(tick,L,time))
        env.plot_body_T(body_name='ur_tcp_link',axis_len=0.1)
        env.plot_body_T(body_name=body_name_clicked,axis_len=0.1)
        if not is_feasible:
            env.plot_sphere(
                p=env.get_p_body(body_name='ur_tcp_link'),r=0.1,rgba=(1,0,0,0.5))
        env.render()
    # Proceed
    if tick < (L-1): tick = tick + 1
    
print ("Done.")
is_point_feasible = partial(
    is_qpos_feasible,
    env              = env,
    joint_names      = joint_names,
    robot_body_names = robot_body_names,
    obj_body_names   = obj_body_names,
    env_body_names   = env_body_names,
) # function of 'qpos'
is_point_to_point_connectable = partial(
    is_qpos_connectable,
    env              = env,
    joint_names      = joint_names,
    robot_body_names = robot_body_names,
    obj_body_names   = obj_body_names,
    env_body_names   = env_body_names,
    deg_th           = 5.0,
) # function of 'qpos1' and 'qpos2'
print ("Ready.")

############################## RRT #################################################
point_min = np.array([-0.5, -2.5, 1.0, 
                      -0.88, -3.14, -0.5])
point_max = np.array([0.5, -1.0, 2.5, 
                      0.88, 3.14, 0.5])
rrt = RapidlyExploringRandomTreesStarClass(
    name      ='RRT-Star-UR',
    point_min = point_min,
    point_max = point_max,
    goal_select_rate = 0.01 ,
    steer_len_max    = np.deg2rad(10),
    search_radius    = np.deg2rad(2), # 10, 30, 50
    norm_ord         = 2, # 2,np.inf,
    n_node_max       = 10000,
    TERMINATE_WHEN_GOAL_REACHED = False, SPEED_UP = True,
)
point_root,point_goal = q0,q_grasp
rrt.init_rrt_star(point_root=point_root,point_goal=point_goal,seed=1)
pos_data = np.zeros((3,1))

while True:
    # Randomly sample a point
    while True:
        doit = np.random.rand()
        if doit <= rrt.goal_select_rate: 
            point_sample = rrt.point_goal
        else:
            point_sample = rrt.sample_point() # random sampling
        check_feas, pos_vec = is_point_feasible(qpos=point_sample)
        if check_feas:
            break
    
    # Get the nearest node ('node_nearest') to 'point_sample' from the tree
    node_nearest = rrt.get_node_nearest(point_sample)
    point_nearest = rrt.get_node_point(node_nearest)

    # Steering towards 'point_sample' to get 'point_new'
    point_new,cost_new = rrt.steer(node_nearest,point_sample)
    if point_new is None: continue # if the steering point is feasible
    feas, feas_pos = is_point_feasible(qpos=point_new)
    if feas and \
        is_point_to_point_connectable(qpos1=point_nearest,qpos2=point_new):
        node_min = node_nearest.copy()
        cost_min = cost_new
        # Select a set of nodes near 'point_new' => 'nodes_near'
        nodes_near = rrt.get_nodes_near(point_new)
        # For all 'node_near' find 'node_min'
        for node_near in nodes_near:
            point_near,cost_near = rrt.get_node_point_and_cost(node_near)
            if is_point_to_point_connectable(qpos1=point_near,qpos2=point_new):
                cost_prime = cost_near + rrt.get_dist(point_near,point_new)
                if cost_prime < cost_min:
                    cost_min = cost_near + rrt.get_dist(point_near,point_new)
                    node_min = node_near
        
        # Add 'node_new' and connect it with 'node_min'
        node_new = rrt.add_node(point=point_new,cost=cost_min,node_parent=node_min)

        # New node information for rewiring
        point_new,cost_new = rrt.get_node_point_and_cost(node_new)

        # Rewire
        for node_near in nodes_near:
            if node_near == 0: continue
            if rrt.get_node_parent(node_near) == node_new: continue
            point_near,cost_near = rrt.get_node_point_and_cost(node_near)
            cost_check = cost_new+rrt.get_dist(point_near,point_new)
            if (cost_check < cost_near) and \
                is_point_to_point_connectable(qpos1=point_near,qpos2=point_new):
                rrt.replace_node_parent(node=node_near,node_parent_new=node_new)

        # Re-update cost of all nodes
        if rrt.SPEED_UP: node_source = node_min
        else: node_source = 0
        rrt.update_nodes_cost(node_source=node_source,VERBOSE=False)

    # Print
    n_node = rrt.get_n_node()
    if (n_node % 1000 == 0) or (n_node == (rrt.n_node_max)):
        cost_goal = rrt.get_cost_goal() # cost to goal
        print ("n_node:[%d/%d], cost_goal:[%.5f]"%
               (n_node,rrt.n_node_max,cost_goal))
    
    # Terminate condition (if applicable)
    if n_node >= rrt.n_node_max: break # max node
    if (rrt.get_dist_to_goal() < 1e-6) and rrt.TERMINATE_WHEN_GOAL_REACHED: break

print(f"shape of pos samples: {pos_data.shape}")
tick = 0
L = pos_data.shape[1]
env.reset()
env.set_state(**state,step=True)

############################## RRT ANIMATION #################################################
# Get joint indices
# from 'start' to the point closest to the 'goal'
node_check = rrt.get_node_nearest(rrt.point_goal)
node_list = [node_check]
while node_check:
    node_parent = rrt.get_node_parent(node_check)
    node_list.append(node_parent)
    node_check = node_parent
node_list.reverse()
print ("node_list:%s"%(node_list))

print("getting joint trajectories")
# Get joint trajectories
q_anchors = np.zeros((len(node_list),len(joint_names)))
for idx,node in enumerate(node_list):
    qpos = rrt.get_node_point(node)
    q_anchors[idx,:] = qpos

print("interpolating")
times_interp,q_interp,_,_ = interpolate_and_smooth_nd(
    anchors   = q_anchors,
    HZ        = env.HZ,
    acc_limit = np.deg2rad(15),
    verbose=True
)
L = len(times_interp)
print ("len(node_list):[%d], L:[%d]"%(len(node_list),L))

print("time to animate!")
# Animate
env.reset()
env.set_state(**state,step=True)
env.init_viewer()
tick = 0
while env.is_viewer_alive():
    # Update
    env.forward(q=q_interp[tick,:],joint_names=joint_names)
    print(q_interp[tick,:])
    # Check collsision 
    is_feasible = is_qpos_feasible(
        env,qpos,joint_names,
        robot_body_names,obj_body_names,env_body_names)
    # Render
    if tick%20 == 0 or tick == (L-1):
        env.plot_text(p=np.array([0,0,1]),label='tick:[%d/%d]'%(tick,L))
        env.render()
    # Increase tick
    if tick < (L-1): tick = tick + 1
print ("Done.")