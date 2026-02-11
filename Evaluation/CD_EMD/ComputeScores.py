"""
Modified from the official PVD implementation:
https://github.com/alexzhou907/PVD
Zhou et al., "Point-Voxel Diffusion for 3D Shape Generation", arXiv:2104.03670
"""

import torch
from pprint import pprint
from metrics.evaluation_metrics import compute_all_metrics
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
import argparse
import json
from pathlib import Path

# Bounding box normalization
def normalization_bb(pcs):
    # pcs: [B, N, 3]
    bbox_min = pcs.min(dim=1, keepdim=True)[0]
    bbox_max = pcs.max(dim=1, keepdim=True)[0]
    bbox_center = (bbox_max + bbox_min) / 2
    bbox_scale = (bbox_max - bbox_min).max(dim=2, keepdim=True)[0] / 2
    pcs_normalized = (pcs - bbox_center) / bbox_scale
    return pcs_normalized

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--sample_pth', type=str, default='', required=True)
ap.add_argument('-r', '--reference_pth', type=str, default='', required=True)
ap.add_argument('-o', '--out_pth', type=str, default='results_metrics_symmetry.json', required=False)
ap.add_argument('-bs', '--batch_size', type=int, default=50, required=False)
ap.add_argument('-n', '--req_norm', type=bool, default=True, required=False)

args = ap.parse_args()

samples_path = args.sample_pth
ref_path = args.reference_pth
output_path = Path(args.out_pth)
batch_size = args.batch_size

# Load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("samples_path:", args.sample_pth)
print("ref_path:", args.reference_pth)
print("output_path:", args.out_pth)
print("batch_size:", args.batch_size)
print("req_norm:", args.req_norm)

sample_data = torch.load(samples_path) # Loads the tensor
sample_pcs = sample_data.contiguous()

ref_data = torch.load(ref_path) # Loads the tensor

if isinstance(ref_data, dict):
    print(f"ref_data is {type(ref_data)}, applying denormalization.")
    ref_pcs = ref_data['ref']
    mean = ref_data['mean'].float()
    std = ref_data['std'].float()

    ref_pcs = ref_pcs * std + mean # Desnormalización

elif torch.is_tensor(ref_data):
    ref_pcs = ref_data.contiguous()

sample_pcs = sample_pcs.float().to(device)
ref_pcs = ref_pcs.float().to(device)

print(sample_pcs.shape)
print(ref_pcs.shape)

# Normalice both sets
if args.req_norm:
    sample_pcs = normalization_bb(sample_pcs.float())
    ref_pcs = normalization_bb(ref_pcs.float())
else:
    sample_pcs = sample_pcs.float()
    ref_pcs = ref_pcs.float()

def stats(name, pcs):
    print(
        f"{name}: "
        f"mean={pcs.mean().item():.4f}, "
        f"std={pcs.std().item():.4f}, "
        f"min={pcs.min().item():.4f}, "
        f"max={pcs.max().item():.4f}"
    )

stats("samples", sample_pcs)
stats("refs", ref_pcs)

print(f"Generation sample size: {sample_pcs.size()} reference size: {ref_pcs.size()}")

# Compute metrics
results = compute_all_metrics(sample_pcs, ref_pcs, batch_size)
results = {k: (v.cpu().detach().item()
                if not isinstance(v, float) else v) for k, v in results.items()}

pprint(results)

# ---- Append to JSON ----
if output_path.exists():
    with output_path.open("r") as f:
        all_results = json.load(f)
else:
    all_results = []

all_results.append(results)

with output_path.open("w") as f:
    json.dump(all_results, f, indent=4)

print(f"[OK] Appended results to {output_path.resolve()}")

#jsd = JSD(sample_pcs.numpy(), ref_pcs.numpy())
#pprint(f'JSD: {jsd}')
#print(f'JSD: {jsd}')

"""
PVD

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/airplane/ckpt_6199/samples_pvd.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/car/ckpt_3299/samples_pvd.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/chair/ckpt_1199/samples_pvd.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm False

LION

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Airplane/generated_pth/samples_lion.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Car/generated_pth/samples_lion.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Chair/generated_pth/samples_lion.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm False

XCube

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/generated_airplane_xcube_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/generated_car_xcube_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/generated_chair_xcube_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True

SLIDE 3D

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/generated_airplane_slide_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/generated_car_slide_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm False

python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/generated_chair_slide_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm False

"""