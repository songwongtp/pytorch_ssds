import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import uuid
import skimage

HOUSE_CLASSES = ( '__background__', # always index 0
        'door', 'garage_door', 'window')

# img_set = ''
# img_set = '_cleaned'

class HouseDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, preproc=None, target_transform=None,
                 dataset_name='House'):
        self.root = root
        self.cache_path = os.path.join(self.root, 'cache')
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name

        self.img_set = '_cleaned' if self.image_set == 'train' else '_new'

        with open(os.path.join(root, "mrcnn_dataset{}.p".format(self.img_set)), "rb") as fileObject:
            self.dataset_processed = pickle.load(fileObject)
        self.ids = list()
        self.annotations = list()
        self._view_map = {
            'train' : [6000, len(self.dataset_processed)],
            'val' : [0, len(self.dataset_processed)],
        }

        for [image_set] in image_sets:
            [start, end] = self._view_map[image_set]
            self.ids.extend(range(start, end))


    def __getitem__(self, index):
        target = self.pull_anno(index)
        
        img = self.pull_image(index)
        height, width, _ = img.shape

        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)
            #print(img.size())

                    # target = self.target_transform(target, width, height)
        #print(target.shape)

        return img, target

    def __len__(self):
        return len(self.ids)
    
    def get_class_names(self):
        return list(HOUSE_CLASSES)

    def image_path_from_index(self, index):
        img_id = self.ids[index]

        img_name = self.dataset_processed[img_id]['img_name']
        #print (img_name, self.dataset_processed[img_id]['URL'])
        image_path = os.path.join(self.root, 'images{}/'.format(self.img_set) + img_name)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        path = self.image_path_from_index(index)
        return cv2.imread(path, cv2.IMREAD_COLOR)

        #pil_image = Image.open(path).convert('RGB') 
        #open_cv_image = np.array(pil_image) 
        
        #image = skimage.io.imread(path)
        # If grayscale. Convert to RGB for consistency.
        #if image.ndim != 3:
        #    image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        #if image.shape[-1] == 4:
        #    image = image[..., :3]
        #open_cv_image = skimage.img_as_ubyte(image)

        # Convert RGB to BGR 
        #return open_cv_image[:, :, ::-1]

    def pull_anno(self, index):
        img_id = self.ids[index]

        coords = self.dataset_processed[img_id]['coords']
        classes = self.dataset_processed[img_id]['classes']

        num_objs = len(coords)
        anno = np.zeros((num_objs, 5))
        for idx, (obj, c) in enumerate(zip(coords, classes)):
            if len(obj) == 0:
                continue

            x1 = min(obj, key = lambda t: t[0])[0]
            y1 = min(obj, key = lambda t: t[1])[1]
            x2 = max(obj, key = lambda t: t[0])[0]
            y2 = max(obj, key = lambda t: t[1])[1]

            anno[idx, 0:4] = np.array([x1, y1, x2, y2])
            anno[idx, 4] = HOUSE_CLASSES.index(c)

        if self.target_transform is not None:
            anno = self.target_transform(anno)
        return anno
    
    def pull_formatted_anno(self, index):
        anno = self.pull_anno(index)
        return anno[:, 0:4], anno[:, 4]

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
    
    def evaluate_detections(self, all_boxes, output_dir=None):
        pass
