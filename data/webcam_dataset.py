from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

import cv2


class WebcamDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

        self.vc = cv2.VideoCapture(0)
        if self.vc.isOpened(): # try to get the first frame
            rval, frame = self.vc.read()
        else:
            raise Exception('Unable to load webcam.')

    def __del__(self):
        self.vc.release()
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        _, frame = self.vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform(Image.fromarray(frame))
        #return {'frame', frame}

        #A_path = 'me.jpg'
        #A_img = Image.open(A_path).convert('RGB')
        #A = self.transform(A_img)
        return {'A': frame, 'A_paths': 'none'}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return 100
