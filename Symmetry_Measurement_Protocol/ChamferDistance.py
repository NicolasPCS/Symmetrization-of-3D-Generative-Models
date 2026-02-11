import numpy as np
from scipy.spatial import KDTree

def chamfer_distance2(pc1, pc2):
    try:
        # Expand dimensions to compute pairwise distances
        a = pc1[:, np.newaxis, :] # (N, 1, 3)
        b = pc2[np.newaxis, :, :] # (1, M, 3)

        # Compute pairwise distances (broadcasted)
        dist = np.linalg.norm(a - b, axis=2) # (N, M)

        # For each point in A -> closest in B
        min_dist_a_to_b = np.min(dist, axis=1)

        # For each point in B -> closest in A
        min_dist_b_to_a = np.min(dist, axis=0)

        # Chamfer distance = average of both
        chamfer = np.mean(min_dist_a_to_b) + np.mean(min_dist_b_to_a)

        return chamfer
    except:
        raise Exception("Error while trying to compute chamfer distance.")

# More scalable to thousand of points
def chamfer_distance(pc1, pc2):
    try:
        tree = KDTree(pc2)
        dist_point_cloud1 = tree.query(pc1)[0]
        tree = KDTree(pc1)
        dist_point_cloud2 = tree.query(pc2)[0]

        chamfer = np.mean(dist_point_cloud1) + np.mean(dist_point_cloud2)

        return chamfer
    except:
        raise Exception("Error while trying to compute chamfer distance.")