import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import yaml
import cv2
import argparse
from tifffile import imwrite
import scipy.io as scpio
from PIL import Image
from skimage.transform import resize
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from scipy import ndimage

# from scipy import ndimage

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME =  'ResizeBilinear_3:0' #'SemanticPredictions:0'

    INPUT_SIZE = 321
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
      """Creates and loads pretrained deeplab model."""
      self.graph = tf.Graph()

      graph_def = None
      # Extract frozen graph from tar archive.
      tar_file = tarfile.open(tarball_path)
      for tar_info in tar_file.getmembers():
        if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
          file_handle = tar_file.extractfile(tar_info)
          graph_def = tf.GraphDef.FromString(file_handle.read())
          break

      tar_file.close()

      if graph_def is None:
        raise RuntimeError('Cannot find inference graph in tar archive.')

      with self.graph.as_default():
          tf.import_graph_def(graph_def, name='')

      logits = self.graph.get_tensor_by_name(self.OUTPUT_TENSOR_NAME)
      self.softmax_output = tf.nn.softmax(logits)

      self.sess = tf.Session(graph=self.graph)
      

    def run(self, image):
      """Runs inference on a single image.

      Args:
        image: A PIL.Image object, raw input image.

      Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
      """
      width, height = image.size      
      resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
      target_image_size = (int(resize_ratio * width), int(resize_ratio * height))
      resized_image = image.convert('RGB').resize(target_image_size, Image.ANTIALIAS)
      
      batch_seg_map = self.sess.run(
          self.softmax_output,
          feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]})
      
      seg_map = batch_seg_map[0]
      sliced_seg_map = seg_map[0:height, 0:width, :]
      return image, sliced_seg_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='TRAIN', help='Mention \'TRAIN\' for training and \'VAL\' for validation set results. Default is \'TRAIN\'.')
    args = parser.parse_args()

    download_cfg = yaml.safe_load(open('semantic_config.yaml'))

    model_path = download_cfg['DATASET']['DOWNLOAD_DIR']
    tar_name = download_cfg["DATASET"]["TARBALL_NAME"]
    download_path = os.path.join(model_path, tar_name) 
    BASE_PATH = download_cfg["SEMANTIC SEG"]["BASE_PATH"]
    RGB_PATH = download_cfg['SEMANTIC SEG']['RGB_PATH']
    SEG_PATH = download_cfg['SEMANTIC SEG']['TRAIN_SEG_RESULT']
    image_path = os.path.join(BASE_PATH, RGB_PATH)

    if args.mode == 'TRAIN':
        IMAGE_LISTS = download_cfg["SEMANTIC SEG"]["TRAIN_LIST"]
    elif args.mode == 'VAL':
	      IMAGE_LISTS = download_cfg["SEMANTIC SEG"]["VAL_LIST"]   
    else:
	      print('Mention correct argument - \'TRAIN\' for training and \'VAL\' for validation set results. Default is \'TRAIN\'.\n')
	      exit()
    
    segment_path = os.path.join(BASE_PATH, SEG_PATH)
    if not os.path.exists(segment_path):
        os.makedirs(segment_path)

    MODEL = DeepLabModel(download_path)

    # Read the training list and load the image names in the lists
    f = open(IMAGE_LISTS,'r')
    image_list = []
    
    for line in f:
        try:
            img_name, _ = line.strip("\n").split(' ')
        except ValueError:
            img_name = _ = line.strip("\n")

        image_list.append(img_name)
    image_list = list(set(image_list))

    # run semantic segmentation on each image and save it.
    # import pdb
    # pdb.set_trace()

    for e, img_name in enumerate(image_list):
        if e%100==0:
            print("Running deeplab on image: "+ str(e+1))
        try:
            image = Image.open(os.path.join(image_path, img_name+download_cfg['SEMANTIC SEG']['RGB_EXT']))
            width, height = image.size
            resized_im, seg_map = MODEL.run(image)           
            imwrite(os.path.join(segment_path, img_name+download_cfg['SEMANTIC SEG']['SEG_EXT']), seg_map, planarconfig='CONTIG')
        except IOError:
		        print("Cannot retrive image ", img_name+download_cfg['SEMANTIC SEG']['RGB_EXT'], ' from path ', os.path.join(image_path, img_name+download_cfg['SEMANTIC SEG']['RGB_EXT']))
        except Exception as e:
            print('Error while processing image: ', img_name+download_cfg['SEMANTIC SEG']['RGB_EXT'])
            print(e)

    sem_count = len([name for name in os.listdir(segment_path) if os.path.isfile(os.path.join(segment_path, name))])
    print('No of images in the semantic folder: ', sem_count)

    img_count = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
    print("No of images in the images folder: ", img_count)

if __name__ == '__main__':
    main()
