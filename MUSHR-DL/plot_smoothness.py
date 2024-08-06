import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu as u_test

data_steering = np.load('steering_log_steering.npy',allow_pickle=True)[:2000]
data_bezier = np.load('steering_log_bezier.npy',allow_pickle=True)[:2000]
data_image = np.load('steering_log_trajectory.npy',allow_pickle=True)[:2000]
data_bezier_05 = np.load('steering_log_bezier_0.5.npy',allow_pickle=True)[:2000]
data_bezier_15 = np.load('steering_log_bezier_1.5.npy',allow_pickle=True)[:2000]
data_human = np.load('steering_log_human.npy',allow_pickle = True)[:2000]

# print(data)
t_steering = np.arange(0,len(data_steering)/15,1/15)
t_steering = t_steering
t_bezier = np.arange(0,len(data_steering)/15,1/15)
t_bezier = t_bezier
t_image = np.arange(0,len(data_steering)/15,1/15)
t_image = t_image

def absolute_rate(x):
    return np.fabs(np.diff(x))*15

mse_steering = np.mean(absolute_rate(data_steering)**2)*6.5
mse_bezier = np.mean(absolute_rate(data_bezier)**2)*6.5
mse_bezier05 = np.mean(absolute_rate(data_bezier_05)**2)*6.5
mse_bezier15 = np.mean(absolute_rate(data_bezier_15)**2)*6.5
mse_image = np.mean(absolute_rate(data_image)**2)*6.5
mse_human = np.mean(absolute_rate(data_human)**2)*6.5

x = absolute_rate(data_image)
y = absolute_rate(data_bezier)
z = absolute_rate(data_steering)
w = absolute_rate(data_human)
u,p1 = u_test(x,y)
u,p2 = u_test(x,z)
u,p3 = u_test(y,z)
u,p4 = u_test(y,w)
u,p5 = u_test(x,z)
print(p1,p2,p3,p4,p5)

# print(mse_steering,mse_image,mse_bezier,mse_bezier05,mse_bezier15,mse_human)

# plt.plot(6.5*t_image[:-2],absolute_rate(data_bezier_05),label='0.5 s horizon')
# plt.plot(6.5*t_steering[:-2],absolute_rate(data_bezier_15),label='1.5 s horizon')
plt.plot(6.5*t_bezier[:-2],absolute_rate(data_image),label='image-image')
plt.plot(6.5*t_bezier[:-2],absolute_rate(data_steering),label='steering')

plt.plot(6.5*t_bezier[:-2],absolute_rate(data_bezier),label='1 s horizon bezier')
plt.plot(6.5*t_bezier[:-2],absolute_rate(data_human),label='human driving')

plt.xlabel('distance (m)')
plt.ylabel('|d(steering)/dt| (deg/s)')
plt.legend()
plt.show()