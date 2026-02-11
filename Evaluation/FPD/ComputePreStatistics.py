import torch
import numpy as np
import argparse

import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from ComputeFPD import load_pcs_from_pth, normalize
from evaluation.pointnet import PointNetCls
from evaluation.FPD import save_statistics

real = load_pcs_from_pth("/home/ncaytuir/TreeGAN_necs/MyScripts/airplane/ckpt_6199/reference.pth")
real = normalize(real)
print("Real pcs shape: ", real.shape)

# Load PointNet
model = PointNetCls(k=16)
model.load_state_dict(torch.load('/home/ncaytuir/TreeGAN_necs/evaluation/cls_model_39.pth', map_location='cpu'))
model.eval()

# Save statistics
save_statistics(real, '/home/ncaytuir/TreeGAN_necs/MyScripts/airplane/ckpt_6199/normalized_gt_pre_statistics.npz', model=model, batch_size=64, dims=1808, cuda=None)
print("Done")