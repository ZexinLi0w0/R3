from pytorch_high_level import *
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
from scipy.ndimage.filters import gaussian_filter
import math as m
from point_warping import *
import argparse
import os

parser = argparse.ArgumentParser(description='IL')
parser.add_argument('--dataset_name', type=str, default="4", help='suffix for the dataset name')
args = parser.parse_args()


HEIGHT = 240
WIDTH = 320
CHANNELS = 1

cam_height = 0.845 # found from 3d model of car (urdf)
X_max = WIDTH #image pixels in horizontal direction
Y_max = HEIGHT #image pixels in vertical direction
X_Center = X_max//2 #assuming image center is at geometric center (camera matrix should be used, but this will do for now)
Y_Center = Y_max//2
pitch = 0
focalLength = 50 #hd1080
fov_h = 90/57.3 #fov 
K_h = m.tan(fov_h/2)
fov_v = 58/57.3
K_v = m.tan(fov_v/2)
K = np.array([[focalLength,   0.        , X_Center],
       [  0.        , focalLength,      Y_Center],
       [  0.        ,   0.        ,   1.        ]])

x_scale = 30
y_scale = 30

model_name = 'trajectory.h5'
checkpoint = torch.load(model_name)
model = checkpoint['net']
model = model.to(device)
print("model loaded")
train_log = []

def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    mask = cv2.bitwise_and(img,mask)
    return mask

roi_vert = np.array([[WIDTH*0.05,HEIGHT*0.99],[WIDTH*0.1,HEIGHT*0.6],[WIDTH*0.9,HEIGHT*0.6],[WIDTH*0.95,HEIGHT*0.99]],dtype=np.int32)
kernel = np.ones((9,9),np.uint8)

output_path = os.getcwd()+"/Outputs_pytorch"
counter = 0

torch.cuda.empty_cache()
train_data = np.load('MUSHR_320x240_shuffled_Image_{}.npy'.format(args.dataset_name),allow_pickle=True)
X = ([i[0] for i in train_data])
Y = ([i[1] for i in train_data])
print("data loaded")
train_data = None
del train_data
with torch.no_grad():
    for j in range(0,len(X),100):
        img = X[j]
        img = torch.Tensor(img).view(-1,1,240,320).to(device)
        expected = Y[j]
        now = time.time()
        traj = model(img)[0].cpu().numpy()
        # print(traj)
        traj = traj.astype(np.uint8)
        traj = traj.reshape((240,320,1))
        ret, traj = cv2.threshold(traj, 0, 255, cv2.THRESH_BINARY)
        traj = roi(traj,[roi_vert])
        traj = cv2.morphologyEx(traj, cv2.MORPH_OPEN, kernel)
        traj = cv2.morphologyEx(traj, cv2.MORPH_CLOSE, kernel)

        added_image = np.zeros((240,320,3))
        added_image[...,:] = X[j].reshape((240,320,1))
        added_image[:,:,1] = added_image[:,:,1] + traj
        added_image[:,:,2] = added_image[:,:,2] + expected*0.5
        index = np.where(traj)
        index = np.array(index)
        top = np.min(index[0])
        bottom = np.max(index[0])
        x = []
        y = []
        for i in reversed(range(top,bottom)):
            ind = np.where(index[0]==i)
            array = index[1,ind]
            pos = np.mean(array)
            if not np.isnan(pos):
                x.append(pos)
            else:
                x.append(x[-1])
            y.append(i)
        x = np.array(x)
        y = np.array(y)
        x = gaussian_filter(x,sigma=10)
        x_m,y_m = img2XY(x, y,cam_height,K_v,K_h,X_max,Y_max,X_Center,Y_Center,pitch)
        dt = time.time()-now
        print(dt*1000)
        pts = np.column_stack((np.int32(x_m*x_scale+X_Center),np.int32(Y_max - y_m*y_scale)))
        cv2.polylines(added_image, [pts], False, 255, 5)
        cv2.imwrite(output_path+"/{}.jpg".format(str(counter)),added_image)
        cv2.imwrite(output_path+"/input{}.jpg".format(str(counter)),X[j])
        counter += 1
print(x,y)
X = None
Y = None
del X
del Y
