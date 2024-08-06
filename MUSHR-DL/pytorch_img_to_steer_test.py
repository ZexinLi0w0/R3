from pytorch_high_level import *
import argparse

parser = argparse.ArgumentParser(description='testing')
parser.add_argument('--dataset_name', type=str, default="4", help='suffix for the dataset name')
args = parser.parse_args()

HEIGHT = 240
WIDTH = 320
CHANNELS = 1
model_name = 'steering.h5'

checkpoint = torch.load(model_name)
model = checkpoint['net']
model = model.to(device)

train_data = np.load('MUSHR_320x240_shuffled_Steering_{}.npy'.format(args.dataset_name),allow_pickle=True)
X = ([i[0] for i in train_data])
Y = ([i[1] for i in train_data])
with torch.no_grad():
    for j in range(0,len(X),10):
        img = np.array(X[j],dtype=np.float32)
        time.sleep(0.001)
        img = torch.Tensor(img).view(-1,1,240,320).to(device)
        # print(img.dtype)
        now = time.time()
        output = model(img)[0]
        delta = time.time()-now
        print(delta*1000)
        output = float(output[0] - output[1])
        GT = Y[j]
        GT = GT[0] - GT[1]
        error = np.fabs(GT-output)
        print(error)
train_data = None
X = None
Y = None
del train_data
del X
del Y
