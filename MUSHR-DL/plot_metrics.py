import numpy as np
import matplotlib.pyplot as plt

data = np.load('trajectory_log_2020_6_5_18_44.npy',allow_pickle=True)
# print(data)
train_loss = data[:,0]
print(train_loss)
val_loss = data[:,1]

t = np.arange(0,len(train_loss),1)

plt.plot(t,train_loss,label='train_loss')
plt.plot(t,val_loss,label='val_loss')
plt.legend()
plt.show()