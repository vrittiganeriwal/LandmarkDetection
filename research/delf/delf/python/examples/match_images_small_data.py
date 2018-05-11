# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Matches two images using their DELF features.

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys,os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf

from tensorflow.python.platform import app
from delf import feature_io

cmd_args = None

_DISTANCE_THRESHOLD = 0.8

def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  # with tf.gfile.GFile(list_path, 'r') as f:
  #   image_paths = f.readlines()
  # image_paths = [entry.rstrip() for entry in image_paths]
  # return image_paths
  image_paths=[]
  for dir, subdir, files in os.walk(list_path):
      for file in files:
          image_paths.append(os.path.join(dir, file))
  return sorted(image_paths)

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Read features for test image
  test_locations, _, test_descriptors, _, _ = feature_io.ReadFromFile(
      cmd_args.test_features_path)
  test_num_features = test_locations.shape[0]
  print("locations and descriptors for test image", test_num_features)

  image_paths = _ReadImageList(cmd_args.list_images_path)
  num_images = len(image_paths)
  tf.logging.info('done! Found %d images', num_images)

  # print(image_paths)

  feature_paths = _ReadImageList(cmd_args.list_images_features_path)
  num_features = len(feature_paths)
  tf.logging.info('done! Found %d images', num_features)

  # print(feature_paths)


  ###############################

  # Read features NEW
  locations_list = []
  descriptors_list = []
  num_features_list = []
  for feature in range(num_features):
      l, _, d, _, _ = feature_io.ReadFromFile(
          feature_paths[feature])
      locations_list.append(l)
      descriptors_list.append(d)
      # print(l.shape[0])
      num_features_list.append(l.shape[0])
      tf.logging.info("Image: %s" % image_paths[feature])
      tf.logging.info("Loaded image %d features" % num_features_list[feature])


  # Find nearest-neighbor matches using a KD tree NEW
  indices_list = []
  d1_tree = cKDTree(test_descriptors)
  for des in range(len(descriptors_list)):
      _, indices = d1_tree.query(
          descriptors_list[des], distance_upper_bound=_DISTANCE_THRESHOLD)
      # if des == 13:
      #   print("Indices %d", (des,indices))
      # print(indices)
      indices_list.append(indices)


  # Select feature locations for putative matches NEW
  locations_to_use_1_list = []
  locations_to_use_2_list = []
  # print(indices_list[0])
  for l in range(len(locations_list)):
      # print("HHIIIII",l)
      locations_2_to_use = np.array([
                                        locations_list[l][i,]
                                        for i in range(num_features_list[l])
                                        if indices_list[l][i] != test_num_features
                                        ])
      locations_1_to_use = np.array([
                                        test_locations[indices_list[l][j],]
                                        for j in range(num_features_list[l])
                                        if indices_list[l][j] != test_num_features
                                        ])
      locations_to_use_1_list.append(locations_1_to_use)
      locations_to_use_2_list.append(locations_2_to_use)

  # Perform geometric verification using RANSAC NEW
  inliers_list = []
  selected_images = []
  selected_images_locations_to_use_1 = []
  selected_images_locations_to_use_2 = []
  for li in range(len(locations_list)):
      # print(locations_to_use_1_list[li])
      # print(locations_to_use_2_list[li])
      if locations_to_use_1_list[li] != [] and locations_to_use_2_list[li] != []:
          # print("inside first if")
          _, inliers = ransac(
              (locations_to_use_1_list[li], locations_to_use_2_list[li]),
              AffineTransform,
              min_samples=3,
              residual_threshold=20,
              max_trials=1000)
          # print("HERE")
          # if inliers.any()==None:
          #     print("No inliners")
          # print(type(inliers))
          if(inliers is not None):
              tf.logging.info("Image: %s" % image_paths[li])
              tf.logging.info('Found %d inliers' % sum(inliers))
              min_features = min(num_features_list[li],test_num_features)
              # print("Min features %d", (li,min_features))
              if sum(inliers) >= 0.1*min_features:
                  inliers_list.append(inliers)
                  selected_images.append(li)
                  selected_images_locations_to_use_1.append(locations_to_use_1_list[li])
                  selected_images_locations_to_use_2.append(locations_to_use_2_list[li])

  print(len(selected_images))
  # Visualize correspondences, and save to file NEW
  img_list = []
  inlier_idxs_list = []
  test_img = mpimg.imread(cmd_args.test_image_path)
  for i in range(len(selected_images)):
      print(image_paths[selected_images[i]])
      img_temp = mpimg.imread(image_paths[selected_images[i]])
      # print(img_temp.shape)
      img_list.append(img_temp)
      inlier_idxs = np.nonzero(inliers_list[i])[0]
      inlier_idxs_list.append(inlier_idxs)

  for p in range(len(img_list)):
      _, ax = plt.subplots()
      # print(p)
      # print(img_list[p].shape)
      plot_matches(ax,test_img,
                   img_list[p],
                   selected_images_locations_to_use_1[p],
                   selected_images_locations_to_use_2[p],
                   np.column_stack((inlier_idxs_list[p], inlier_idxs_list[p])),
                   matches_color='b')
      ax.axis('off')
      ax.set_title('DELF correspondences')
      position = image_paths[p].rfind('/')

      plt.savefig(cmd_args.output_image + image_paths[selected_images[p]][position:])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--test_image_path',
      type=str,
      default='test_images/image_1.jpg',
      help="""
      Path to test image.
      """)
  parser.add_argument(
      '--test_features_path',
      type=str,
      default='test_features/image_1.delf',
      help="""
      Path to DELF features from test image.
      """)
  parser.add_argument(
      '--list_images_path',
      type=str,
      default='list_images.txt',
      help="""
        Path to list of images.
        """)
  parser.add_argument(
      '--list_images_features_path',
      type=str,
      default='list_images.txt',
      help="""
        Path to list of DELF features.
        """)
  # parser.add_argument(
  #     '--image_2_path',
  #     type=str,
  #     default='test_images/image_2.jpg',
  #     help="""
  #     Path to test image 2.
  #     """)

  # parser.add_argument(
  #     '--features_2_path',
  #     type=str,
  #     default='test_features/image_2.delf',
  #     help="""
  #     Path to DELF features from image 2.
  #     """)
  parser.add_argument(
      '--output_image',
      type=str,
      default='test_match.png',
      help="""
      Path where an image showing the matches will be saved.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
