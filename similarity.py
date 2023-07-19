import os

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh


def compute_trimesh_chamfer(gt_points, gen_mesh, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    # gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


folder_path = "out/pointcloud/room_3plane/generation_pretrained/input/rooms_04"

# # ground_truth_samples_filename = "data/synthetic_room_dataset/rooms_04/00000001/pointcloud0.ply"
# ground_truth_samples_filename = "out/pointcloud/diff_3plane_2/generation_pretrained/input/rooms_04/00000001.ply"
generated_mesh_filename = "out/pointcloud/final_vae_std_1_weight_1_new/generation/meshes/rooms_04/00000319.off"
gen_mesh = trimesh.load(generated_mesh_filename)
# ground_truth_points = trimesh.load(ground_truth_samples_filename)
chamfer_dist_list = []

# print(chamfer_dist)

for i in range(1, 301):
    file_index = str(i).zfill(8)

    # Create the filename by combining the folder path and the formatted file index
    file_name = os.path.join(folder_path, file_index + ".ply")

    ground_truth_points = trimesh.load(file_name)
    chamfer_dist = compute_trimesh_chamfer(ground_truth_points, gen_mesh)
    chamfer_dist_list.append((chamfer_dist, file_index))


chamfer_dist_list.sort()
for chamfer_dist, file_index in chamfer_dist_list[:5]:
    print("File Index:", file_index)
    print("Chamfer Dist:", chamfer_dist)
    print("----------")
    # Check if the file exists before processing it
    # if os.path.isfile(file_name):
    #     # Process the file here
    #     print("Processing file:", file_name)
    # else:
    #     print("File not found:", file_name)
