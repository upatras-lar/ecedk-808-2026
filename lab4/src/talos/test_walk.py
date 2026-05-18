import numpy as np
from talos import Talos

robot =  Talos()

traj = robot.march(n_steps =  5,
                   travel_distance = 1.,
                   time_step = 1.,
                   theta = 0.)

robot.visualize_traj(traj)