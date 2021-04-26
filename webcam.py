"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import tensor2im
import torchvision.transforms as transforms

import time
import cv2
import numpy as np

opt = TestOptions().parse()
#--dataroot ./datasets/selfies --direction AtoB --model pix2pix --name selfies_pix2pix --num_test 400 --gpu_ids -1
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

opt.dataset_mode = 'webcam'

dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

window_name = 'Press Esc to cancel'
cv2.namedWindow(window_name)

tic = time.perf_counter()
count = 0
while True:
    data = next(iter(dataset))
    count += 1
    model.real_A = data['A'].to(model.device)
    model.forward()
    raw = tensor2im(model.real_A)
    fake = tensor2im(model.fake_B)
    fake = ((fake.astype(np.float16) + raw.astype(np.float16))/2).astype(fake.dtype)
    output_image = np.concatenate((raw, fake), axis=1)
    cv2.imshow(window_name, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR) )
    key = cv2.waitKey(10)
    if key == 27: break

toc = time.perf_counter()
seconds = toc - tic

print(f'Performed {count} enhancements in {seconds:0.1f}s')
print(f'{count/seconds:0.1f} enhancements/sec')
print(f'ms/enhancement = {1000*seconds//count}')

cv2.destroyWindow(window_name)
