# Copyright (c) 2019, Grégoire Payen de La Garanderie, Durham University
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
# from pyson.utils import multi_thread
from avcv.process import multi_thread
import mmcv
from glob import glob
from tqdm import tqdm
import json
import numpy as np
import math
import cv2
import io

import os
import sys

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils

import matplotlib.cm
cmap = matplotlib.cm.get_cmap("viridis")




def get_3d_bbox(camera_calibration, label):
    """
        Return the 8 points bbox, each point has three number (x,y,z)
        To get the coressponding position on the image with perspective transform->, x/z, y/z

    """
    box_to_image = get_box_to_image(camera_calibration, label)
    vertices = np.empty([2,2,2,3])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = v #[v[0]/v[2], v[1]/v[2]]
    return vertices

def get_box_to_image(camera_calibration, label):
    def get_box_transformation_matrix(box):
        """Create a transformation matrix for a given label box pose."""

        tx,ty,tz = box.center_x,box.center_y,box.center_z
        c = math.cos(box.heading)
        s = math.sin(box.heading)

        sl, sh, sw = box.length, box.height, box.width

        return np.array([
            [ sl*c,-sw*s,  0,tx],
            [ sl*s, sw*c,  0,ty],
            [    0,    0, sh,tz],
            [    0,    0,  0, 1]])

    vehicle_to_image = utils.get_image_transform(camera_calibration)
    box_to_vehicle = get_box_transformation_matrix(label.box)
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)
    return box_to_image

def get_annotations(camera_calibration, labels, visibility):

    # Get the transformation matrix from the vehicle frame to image space.
    # vehicle_to_image = utils.get_image_transform(camera_calibration)
#
    # Draw all the groundtruth labels
    list_points = []
    for label, vis in zip(labels, visibility):
        vertex = get_3d_bbox(camera_calibration, label)
        list_points.append({'vertex': vertex, 'label': label, 'vis': vis})
    return list_points

def display_labels_on_image(camera_calibration, img, labels, visibility):

    # Get the transformation matrix from the vehicle frame to image space.
    vehicle_to_image = utils.get_image_transform(camera_calibration)

    # Draw all the groundtruth labels
    for label, vis in zip(labels, visibility):
        if vis:
            colour = (0, 0, 200)
        else:
            colour = (128, 0, 0)
        utils.draw_3d_box(img, vehicle_to_image, label, colour=colour)


def display_laser_on_image(img, pcl, pcl_attr, vehicle_to_image):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl, np.ones_like(pcl[:, 0:1])), axis=1)

    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1)

    # Filter LIDAR points which are behind the camera.
    mask = proj_pcl[:, 2] > 0
    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = pcl_attr[mask]

    # Project the point cloud onto the image.
    proj_pcl = proj_pcl[:, :2]/proj_pcl[:, 2:3]

    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(proj_pcl[:, 0] > 0, proj_pcl[:, 0] < img.shape[1]),
        np.logical_and(proj_pcl[:, 1] > 0, proj_pcl[:, 1] < img.shape[1]))

    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = proj_pcl_attr[mask]
    # Colour code the points based on distance.
    depth = np.concatenate([proj_pcl, proj_pcl_attr[:, :1]], 1)# xy,v
    return depth

def visualize_depth(img, depth):
    proj_pcl, proj_pcl_attr = depth[:,:1], depth[:,1:]
    coloured_intensity = 255*cmap(proj_pcl_attr/30)
    depth_map = np.zeros(img.shape[:2])
    # Draw a circle for each point.
    for i in range(proj_pcl.shape[0]):
        x,y = (int(proj_pcl[i, 0]), int(proj_pcl[i, 1]))
        cv2.circle(img, (x,y), 1, coloured_intensity[i])
        if y < depth_map.shape[0] and x<depth_map.shape[1]:
            depth_map[y,x] = proj_pcl_attr[i]
    return depth_map

# def display_laser_on_image(img, pcl, pcl_attr, vehicle_to_image):
#     # Convert the pointcloud to homogeneous coordinates.
#     pcl1 = np.concatenate((pcl, np.ones_like(pcl[:, 0:1])), axis=1)

#     # Transform the point cloud to image space.
#     proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1)

#     # Filter LIDAR points which are behind the camera.
#     mask = proj_pcl[:, 2] > 0
#     proj_pcl = proj_pcl[mask]
#     proj_pcl_attr = pcl_attr[mask]

#     # Project the point cloud onto the image.
#     proj_pcl = proj_pcl[:, :2]/proj_pcl[:, 2:3]

#     # Filter points which are outside the image.
#     mask = np.logical_and(
#         np.logical_and(proj_pcl[:, 0] > 0, proj_pcl[:, 0] < img.shape[1]),
#         np.logical_and(proj_pcl[:, 1] > 0, proj_pcl[:, 1] < img.shape[1]))

#     proj_pcl = proj_pcl[mask]
#     proj_pcl_attr = proj_pcl_attr[mask]

#     # Colour code the points based on distance.
#     coloured_intensity = 255*cmap(proj_pcl_attr[:, 0]/30)

#     # Draw a circle for each point.
#     for i in range(proj_pcl.shape[0]):
#         cv2.circle(img, (int(proj_pcl[i, 0]), int(
#             proj_pcl[i, 1])), 1, coloured_intensity[i])


if len(sys.argv) != 2:
    print("""Usage: python display_laser_on_image.py <datafile>
Display the groundtruth 3D bounding boxes and LIDAR points on the front camera video stream.""")
    sys.exit(0)

    # Open a .tfrecord
    # filename = sys.argv[1]


def extract_tf_file(filename):
    datafile = WaymoDataFileReader(filename)

    # Generate a table of the offset of all frame records in the file.
    table = datafile.get_record_table()

    print("There are %d frames in this file." % len(table))
    # Loop through the whole file
    # and display 3D labels.
    img_prefex = './data/images'

    os.makedirs(img_prefex, exist_ok=True)
    name = os.path.basename(filename).split('.')[0]
    annotation_path = f'./data/annotations/train_{name}.json'
    os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
    images = []
    annotations = []
    for frameno, frame in tqdm(enumerate(datafile), total=len(table)):

        # Get the top laser information
        laser_name = dataset_pb2.LaserName.TOP
        laser = utils.get(frame.lasers, laser_name)
        laser_calibration = utils.get(
            frame.context.laser_calibrations, laser_name)

        # Parse the top laser range image and get the associated projection.
        ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(
            laser)

        # Convert the range image to a point cloud.
        pcl, pcl_attr = utils.project_to_pointcloud(
            frame, ri, camera_projection, range_image_pose, laser_calibration)

        # Get the front camera information
        camera_name = dataset_pb2.CameraName.FRONT
        camera_calibration = utils.get(
            frame.context.camera_calibrations, camera_name)
        camera = utils.get(frame.images, camera_name)
        out_img_path = os.path.join(
            img_prefex, f'{name}_{camera_name}_{frameno}.jpg')

        # Get the transformation matrix for the camera.
        vehicle_to_image = utils.get_image_transform(camera_calibration)

        # Decode the image
        img = utils.decode_image(camera)
        # Save to disk

        height, width = img.shape[:2]
        images.append(dict(
            file_name=os.path.basename(out_img_path), height=height, width=width, id=len(images)
        ))
        image_id = images[-1]['id']

        # BGR to RGB
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_img_path, img)

        # Some of the labels might be fully hidden therefore we attempt to compute the label visibility
        # by counting the number of LIDAR points inside each label bounding box.

        # For each label, compute the transformation matrix from the vehicle space to the box space.
        vehicle_to_labels = [np.linalg.inv(utils.get_box_transformation_matrix(
            label.box)) for label in frame.laser_labels]
        vehicle_to_labels = np.stack(vehicle_to_labels)

        # Convert the pointcloud to homogeneous coordinates.
        pcl1 = np.concatenate((pcl, np.ones_like(pcl[:, 0:1])), axis=1)

        # Transform the point cloud to the label space for each label.
        # proj_pcl shape is [label, LIDAR point, coordinates]
        proj_pcl = np.einsum('lij,bj->lbi', vehicle_to_labels, pcl1)

        # For each pair of LIDAR point & label, check if the point is inside the label's box.
        # mask shape is [label, LIDAR point]
        mask = np.logical_and.reduce(np.logical_and(
            proj_pcl >= -1, proj_pcl <= 1), axis=2)

        # Count the points inside each label's box.
        counts = mask.sum(1)

        # Keep boxes which contain at least 10 LIDAR points.
        visibility = counts > 10

        # Display the LIDAR points on the image.
        depth_map = display_laser_on_image(img, pcl, pcl_attr, vehicle_to_image)
        # Display the label's 3D bounding box on the image.
        out_depth_path = out_img_path.replace('/images/', '/depth/').replace('.jpg', '')
        # mmcv.imwrite((depth_map*1000).astype(np.uint16), out_depth_path)
        mmcv.mkdir_or_exist(os.path.dirname(out_depth_path))
        np.save(out_depth_path, depth_map)
        anns = get_annotations(
            camera_calibration, frame.laser_labels, visibility)
        bbox_3d = np.array([_['vertex'] for _ in anns if _['vertex'] is not None])
        for ann in anns:
            annotations.append(
                dict(
                    bbox_3d=ann['vertex'].tolist(
                    ) if ann['vertex'] is not None else None,
                    image_id=image_id,
                    category_id=int(ann['label'].type),
                    id=len(annotations)
                )
            )

    # multi_thread(f, enumerate(datafile), verbose=True)
    categories = [
        {
            "id": 2,
            "name": "PEDESTRIAN"
        },
        {
            "id": 4,
            "name": "CYCLIST"
        },
        {
            "id": 3,
            "name": "SIGN"
        },
        {
            "id": 0,
            "name": "UNKNOWN"
        },
        {
            "id": 1,
            "name": "VEHICLE"
        }
    ]

    with open(annotation_path, 'w') as f:
        json.dump(dict(
            images=images,
            annotations=annotations,
            categories=categories
        ), f)

    print('annotation_path:', annotation_path)


if len(sys.argv) != 2:
    print("""Usage: python display_laser_on_image.py <datafile>
Display the groundtruth 3D bounding boxes and LIDAR points on the front camera video stream.""")
    sys.exit(0)
# Open a .tfrecord

def get_cmd(path):
    return "python examples/extract_3d_data.py {}".format(path)

filename = sys.argv[1]
if os.path.isdir(filename):
    filenames = glob(os.path.join(filename, '*.tfrecord'))
    num_process = 10
    nprocess = 0
    prev_is_enter = False
    s = ""
    for i, filename in enumerate(filenames):
        cmd = get_cmd(filename)
        nprocess+=1


        if not prev_is_enter and i!=0:
            cmd = " | "+cmd
        else:
            cmd = cmd
            prev_is_enter = False
        s += cmd
        if nprocess == num_process:
            s+="\n"
            prev_is_enter = True
            nprocess = 0


    with open('cmd.sh','w') as f:
        f.write(s)

    #     if i % num_process == 0 and i!=0:
    #         cmd += "\n" 
    #     else:
    #         cmd += " | \t"
    #     s += cmd
    print(s)

        # extract_tf_file(filename)

else:
    extract_tf_file(filename)
