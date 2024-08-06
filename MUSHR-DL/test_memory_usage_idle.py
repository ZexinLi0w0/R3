import torch
import torch.nn as nn
import models
import argparse

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--testing', type=bool, default=False, help='test mode')
args = parser.parse_args()

model = models.get_model('dave2')
inputs = torch.randn(args.batch_size, 1, 240, 320)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

if args.testing:
    # simulate the test loop
    model.eval()
    for i in range(1000):
        outputs = model(inputs)
        print(f"Batch {i} done")
else:
    # simulate the training loop
    model.train()
    for i in range(1000):
        outputs = model(inputs)
        loss = loss_function(outputs, outputs) # simulate the loss
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        print(f"Batch {i} done")

max_memory = torch.cuda.max_memory_allocated() / 1024**3
print(f"Max memory used by torch: {max_memory:.2f} GB")