"""
Call as: python Symmetry.py ../ShapeNetCore.v3.PC15k/02691156/train chamfer_results_airplane.json npy
"""

import numpy as np
import argparse
import json
import torch

from Householder_transform import householder_transformation
from ChamferDistance import chamfer_distance
from FarthestPointSampling import farthest_point_sampling
from pathlib import Path

# Argument parser
parser = argparse.ArgumentParser(description="Compute symmetry")
parser.add_argument("path", type=str, help="Path to the directory")
parser.add_argument("path_save", type=str, default='chamfer_results.json', help='Path to save results (ext .json)')
parser.add_argument("file_ext", type=str, default='npy', help='File extension (i.e., .npy)')

args = parser.parse_args()
path = Path(args.path)
path_to_save = args.path_save
file_ext = args.file_ext

print(path_to_save)

def is_centered_x(points, tolerance=1e-3):
    centroid_x = np.mean(points[:, 0])
    return abs(centroid_x) < tolerance, centroid_x

def center_in_x(points, centroid_x):
    points[:, 0] -= centroid_x
    return points

def compute_symmetry(dir_path, dest_path, file_ext):
    if file_ext == 'npy':
        files = list(dir_path.glob("*.npy"))

        dictionary = compute_files(files, file_ext)

        # Save dictionary
        with open(dest_path, 'w') as file:
            json.dump(dictionary, file, indent=4)

        return
    elif file_ext == 'pkl':
        files = list(dir_path.glob("*.pkl"))

        dictionary = compute_files(files, file_ext)

        # Save dictionary
        with open(dest_path, 'w') as file:
            json.dump(dictionary, file, indent=4)

        return
    elif file_ext == 'xyz':
        files = list(dir_path.glob("*.xyz"))

        dictionary = compute_files(files, file_ext)

        # Save dictionary
        with open(dest_path, 'w') as file:
            json.dump(dictionary, file, indent=4)

        return
    else:
        return "Invalid file extension."
    
def compute_files(files, file_ext):
    dictionary = {}
    cont = 1

    for p in files:
        print(f'Computing {cont}: {p.name}')

        if file_ext == 'npy':
            # Load point cloud
            points = np.load(p)

            is_centered, cx = is_centered_x(points)

            if is_centered:
                print(f"Centered in X: {is_centered} (centroid X = {cx:.5f})")
                all_original_points = points
            else:
                centered_points = center_in_x(points, cx)
                after_is_centered, after_cx = is_centered_x(centered_points)

                print(f"Centered in X: {after_is_centered} (centroid X = {after_cx:.5f})")
                all_original_points = centered_points

            # Compute FPS
            if all_original_points.shape[0] > 2048:
                original_points = farthest_point_sampling(all_original_points, 2048)
            else:
                original_points = all_original_points

        elif file_ext == 'pkl':
            # Load point cloud
            data = torch.load(p, map_location='cpu')
            points = data['fine_xyz']

            is_centered, cx = is_centered_x(points)

            if is_centered:
                print(f"Centered in X: {is_centered} (centroid X = {cx:.5f})")
                all_original_points = points
            else:
                centered_points = center_in_x(points, cx)
                after_is_centered, after_cx = is_centered_x(centered_points)

                print(f"Centered in X: {after_is_centered} (centroid X = {after_cx:.5f})")
                all_original_points = centered_points

            # Compute FPS
            if all_original_points.shape[0] > 2048:
                original_points = farthest_point_sampling(all_original_points, 2048)
            else:
                original_points = all_original_points

        elif file_ext == 'xyz':
            # Load point cloud
            data = np.loadtxt(p)
            points = data[:, :3]

            is_centered, cx = is_centered_x(points)

            if is_centered:
                print(f"Centered in X: {is_centered} (centroid X = {cx:.5f})")
                all_original_points = points
            else:
                centered_points = center_in_x(points, cx)
                after_is_centered, after_cx = is_centered_x(centered_points)

                print(f"Centered in X: {after_is_centered} (centroid X = {after_cx:.5f})")
                all_original_points = centered_points
            
            # Compute FPS
            if all_original_points.shape[0] > 2048:
                original_points = farthest_point_sampling(all_original_points, 2048)
            else:
                original_points = all_original_points

        # Householder transformation
        transformed_points = householder_transformation(original_points)

        # Compute Chamfer distance
        cd = chamfer_distance(original_points, transformed_points)

        print(f"CD result: {cd} - over {transformed_points.shape} points (before {all_original_points.shape})")

        # Save results in a dictionary
        dictionary.update({p.name: cd})

        cont += 1

    return dictionary

compute_symmetry(path, path_to_save, file_ext)