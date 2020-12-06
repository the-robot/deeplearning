# disable tensorflow debugging messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from unet import UNet

if __name__ == "__main__":
    model = UNet(input_shape = (256, 256, 3))
    model.summary()
