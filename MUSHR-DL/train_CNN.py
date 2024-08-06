from pytorch_high_level import *
# from pytorch_model import Net
import models
import torch
import numpy as np
import time
import argparse
from skimage.transform import resize
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--dataset_name', type=str, default="training", help='dataset name')
parser.add_argument('--model_name', type=str, default="dave2", help='model name')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--test', type=bool, default=False, help='test mode')
parser.add_argument('--model_path', type=str, default=None, help='path to model')

args = parser.parse_args()

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def train(model,
          X,
          Y,
          train_log,
          validation_set=0.1,
          batch_size=32,
          epochs=100,
          model_name='model'):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    data_size = len(X)
    val_size = int(validation_set * len(X))
    train_size = data_size - val_size
    elasped_time = 0

    val_x = X[train_size:]
    val_y = Y[train_size:]
    train_x = X[:train_size]
    train_y = Y[:train_size]

    train_dataset = MyDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = MyDataset(val_x, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(epochs):
        # train
        model.train()
        start_time = time.time()
        train_loss = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= train_size
        end_time = time.time()
        elasped_time += (end_time - start_time)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch in val_dataloader:
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                val_loss += loss.item()
            val_loss /= val_size

        torch.save(model.state_dict(), "./batch_{}/{}_epoch_{}.pth".format(str(batch_size), model_name, epoch))
        print('epoch: {}, train loss: {}, val loss: {}, time: {}'.format(epoch, train_loss, val_loss, elasped_time))
        train_log.append([train_loss, val_loss, elasped_time])

    return train_log

def test(model, X, Y):
    model.eval()
    predictions = []
    with torch.no_grad():
        average_error = 0
        for i in range(X.shape[0]):
            predictions.append(model(X[i].unsqueeze(0)))
        for i in range(len(predictions)):
            average_error += abs(Y[i] - predictions[i][0])
        print('average error: {}'.format(average_error/len(predictions)))
        print(predictions)

HEIGHT = 240
WIDTH = 320
CHANNELS = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = time.localtime(time.time())
assert args.model_name in ['dave2', 'resnet'], "invalid model name"
model_name = 'steering_log_{}_{}_{}_{}_{}_{}.h5'.format(args.model_name, a.tm_year,a.tm_mon,a.tm_mday,a.tm_hour,a.tm_min)

model = models.get_model('dave2').to(device)
print(model)

train_log = []
torch.cuda.empty_cache()
<<<<<<< Updated upstream
X, Y = torch.load('data.pt')
=======
train_data = np.load('MUSHR_320x240_{}.npy'.format(args.dataset_name),allow_pickle=True)

X = torch.Tensor([i[0] for i in train_data]).unsqueeze(1).to(device) # [n, 1, 90, 160]
X = F.interpolate(X, size=(240, 320), mode='bilinear', align_corners=False) # [n, 1, 240, 320]
X /= 255.0
X -= 0.5

Y = torch.Tensor([i[3][4] for i in train_data])\
    .unsqueeze(-1)\
    .to(device) # [n] -> [n,1]
>>>>>>> Stashed changes

if args.test:
    if args.model_path is None:
        print("model path not specified")
        exit(1)
<<<<<<< Updated upstream
    model.load_state_dict(torch.load(args.model_path))
    test(model, X, Y)
    exit(0)
=======
    predictions = []
    model.eval()
    with torch.no_grad():
        average_error = 0
        for i in range(X.shape[0]):
            predictions.append(model(X[i].unsqueeze(0)))
        for i in range(len(predictions)):
            average_error += abs(Y[i] - predictions[i][0])
        print('average error: {}'.format(average_error/len(predictions)))
        exit(0)
>>>>>>> Stashed changes

stored_model_name = 'steering_model_{}_{}_{}_{}_{}'.format(a.tm_year,a.tm_mon,a.tm_mday,a.tm_hour,a.tm_min)
log_name = 'steering_log_{}_{}_{}_{}_{}.npy'.format(a.tm_year,a.tm_mon,a.tm_mday,a.tm_hour,a.tm_min)
os.makedirs('./batch_{}'.format(str(args.batch_size)), exist_ok=True)
train_log = train(model,X,Y,train_log,validation_set=0.1,batch_size=args.batch_size,epochs=args.epochs,model_name=stored_model_name)

# train_log = fit(model,X,Y,train_log,validation_set=0.1,BATCH_SIZE=args.batch_size,EPOCHS=args.epochs,model_name=stored_model_name)
np.save("./batch_{}/{}".format(str(args.batch_size), log_name),np.array(train_log))