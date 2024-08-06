from . import dave2
from . import resnet

# Define a get_model() function that returns a PyTorch model object based on the model name
def get_model(model_name):
    if model_name == 'dave2':
        return dave2.NVIDIA_Dave2()
    elif model_name == 'resnet':
        return resnet.ResNet()
    else:
        raise ValueError('Invalid model name: {}'.format(model_name))
