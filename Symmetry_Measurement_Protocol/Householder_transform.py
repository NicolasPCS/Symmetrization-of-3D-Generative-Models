import numpy as np
import argparse

def householder_transformation(pc):
    try:
        # Define the vector "v" for the YZ plane - Transformation in X
        v = np.array([1,0,0])
        
        # Create the Householder matrix
        H = np.eye(3) - 2 * np.outer(v,v) # Entity matrix

        # Apply transformation
        transformed_points = pc @ H.T # (N, 3) x (3, 3) - Matrix product

        # Save result
        #np.save("archivoreflejado.npy", transformed_points)

        return transformed_points # Return the transformed points

    except:
        raise Exception("Error while trying to do the householder transformation.")