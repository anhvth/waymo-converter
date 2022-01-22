# !pip install addict pandas plyfile tqdm sklearn open3d
import open3d as o3d
import numpy as np
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser("python tools/vis_cloud_point.py $FILE")
    parser.add_argument('file')
    args = parser.parse_args()
    pcd = o3d.geometry.PointCloud()
    points = np.load(args.file)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
