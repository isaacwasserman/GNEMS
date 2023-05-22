import numpy as np
from glob import glob
from PIL import Image
import sklearn.metrics
from skimage.metrics import variation_of_information
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import threading
import os
import json
from PIL.PngImagePlugin import PngInfo

def calculate_overlap(r1, r2):
    # intersection
    a = np.count_nonzero(r1 * r2)
    # union
    b = np.count_nonzero(r1 + r2)
    return a/b

def calculate_SC(segmentation, gt):
    N = segmentation.shape[0] * segmentation.shape[1]
    maxcoverings_sum = 0
    for label1 in np.unique(segmentation):
        # region1 is where the segmentation has a specific label
        region1 = (segmentation == label1).astype(int) 
        # |R| is the size of nonzero elements as this the region size
        len_r = np.count_nonzero(region1) 
        max_overlap = 0
        # Calculate max overlap 
        for label2 in np.unique(gt):
            # region2 is where the segmentation has a specific label
            region2 = (gt == label2).astype(int)
            # Calculate overlap
            overlap = calculate_overlap(region1, region2)
            max_overlap = max(max_overlap, overlap)
        maxcoverings_sum += (len_r * max_overlap)
    return (1 / N) * maxcoverings_sum

def calculate_VI(segmentation, gt):
    VI = np.min(variation_of_information(segmentation, gt))
    return VI

def calculate_PRI(segmentation, gt):
    PRI = sklearn.metrics.adjusted_rand_score(segmentation.flatten(), gt.flatten())
    return PRI

def smart_jaccard(gt_, seg_):
    gt = torch.tensor(gt_)
    seg = torch.tensor(seg_)
    gt_segments = torch.unique(gt)
    seg_segments = torch.unique(seg)
    # assign each float value to an arbitrary int value
    for i, gt_segment in enumerate(gt_segments):
        gt[gt == gt_segment] = i
    for i, seg_segment in enumerate(seg_segments):
        seg[seg == seg_segment] = i
    gt = gt.int()
    seg = seg.int()
    gt_segments = torch.unique(gt)
    seg_segments = torch.unique(seg)
    segment_mapping = {}
    for seg_segment in seg_segments:
        overlaps = torch.zeros(gt_segments.shape[0])
        for i, gt_segment in enumerate(gt_segments):
            overlaps[i] = torch.sum((gt == gt_segment) & (seg == seg_segment))
        segment_mapping[int(seg_segment)] = gt_segments[torch.argmax(overlaps)]
    jaccard = 0
    label_aligned_seg = torch.zeros(seg.shape, dtype=torch.int)
    for seg_segment in seg_segments:
        label_aligned_seg[seg == seg_segment] = segment_mapping[int(seg_segment)]
    for gt_segment in gt_segments:
        jaccard += torch.sum((gt == gt_segment) & (label_aligned_seg == gt_segment)) / torch.sum((gt == gt_segment) | (label_aligned_seg == gt_segment))
    score = float(jaccard / gt_segments.shape[0])
    return score

def smart_f1_score(gt_, seg_):
    gt = torch.tensor(gt_)
    seg = torch.tensor(seg_)
    gt_segments = torch.unique(gt)
    seg_segments = torch.unique(seg)
    # assign each float value to an arbitrary int value
    for i, gt_segment in enumerate(gt_segments):
        gt[gt == gt_segment] = i
    for i, seg_segment in enumerate(seg_segments):
        seg[seg == seg_segment] = i
    gt = gt.int()
    seg = seg.int()
    gt_segments = torch.unique(gt)
    seg_segments = torch.unique(seg)
    segment_mapping = {}
    for seg_segment in seg_segments:
        overlaps = torch.zeros(gt_segments.shape[0])
        for i, gt_segment in enumerate(gt_segments):
            overlaps[i] = torch.sum((gt == gt_segment) & (seg == seg_segment))
        segment_mapping[int(seg_segment)] = int(gt_segments[torch.argmax(overlaps)])
    jaccard = 0
    label_aligned_seg = torch.zeros(seg.shape, dtype=torch.int)
    for seg_segment in seg_segments:
        label_aligned_seg[seg == seg_segment] = segment_mapping[int(seg_segment)]
    return sklearn.metrics.f1_score(gt.flatten(), label_aligned_seg.flatten(), average="weighted")


def calculate_mIOU(segmentation, gt):
    return smart_jaccard(segmentation, gt)

def calculate_f1(segmentation, gt):
    return smart_f1_score(gt, segmentation)

def str_is_num(s):
    return np.array([(char in "0123456789.-") for char in s]).all()

def has_been_evaluated(segmentation):
    conditions1 = np.array([
        "sc" in segmentation.text,
        "vi" in segmentation.text,
        "pri" in segmentation.text,
        "miou" in segmentation.text,
        "f1" in segmentation.text,
    ])
    if not conditions1.all():
        return False
    else:
        return True
        
results = {}
def evaluate(method_name, use_cache=True):
    print(f"Evaluating {method_name}{' without cache' if not use_cache else ''}...")
    with open("../datasets/noise/test_ids.txt", "r") as f:
        noise_segmentation_paths = [f"../results/{method_name}/noise/" + line.strip() + ".png" for line in f.readlines()]
    with open("../datasets/clouds/test_ids.txt", "r") as f:
        cloud_segmentation_paths = [f"../results/{method_name}/clouds/" + line.strip() + ".png" for line in f.readlines()]
    with open("../datasets/texture/test_ids.txt", "r") as f:
        texture_segmentation_paths = [f"../results/{method_name}/texture/" + line.strip() + ".png" for line in f.readlines()]
    # segmentation_times = []
    cloud_scs = []
    cloud_vis = []
    cloud_pris = []
    cloud_mious = []
    cloud_f1s = []
    print(f"Evaluating {method_name} on clouds dataset (1/3)...")
    for path in cloud_segmentation_paths:
        id = path.split("/")[-1].split(".")[0]
        if not os.path.exists(path):
            print(f"Segmentation {path} does not exist.")
            cloud_scs.append(0)
            cloud_vis.append(1)
            cloud_pris.append(0)
            cloud_mious.append(0)
            cloud_f1s.append(0)
            continue
        segmentation = Image.open(path)
        if has_been_evaluated(segmentation) and use_cache:
            sc = float(segmentation.text["sc"])
            vi = float(segmentation.text["vi"])
            pri = float(segmentation.text["pri"])
            miou = float(segmentation.text["miou"])
            f1 = float(segmentation.text["f1"])
        else:
            segmentation = np.array(segmentation)
            gt = np.array(Image.open(f"../datasets/clouds/gt/{id}.png"))
            segmentation = torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0)
            segmentation = torch.nn.functional.interpolate(segmentation, size=gt.shape, mode="nearest").squeeze().numpy()
            sc = calculate_SC(segmentation, gt)
            vi = calculate_VI(segmentation, gt)
            pri = calculate_PRI(segmentation, gt)
            miou = calculate_mIOU(segmentation, gt)
            f1 = calculate_f1(segmentation, gt)
            metadata = PngInfo()
            metadata.add_text("sc", str(sc))
            metadata.add_text("vi", str(vi))
            metadata.add_text("pri", str(pri))
            metadata.add_text("miou", str(miou))
            metadata.add_text("f1", str(f1))
            Image.fromarray(segmentation).save(path, pnginfo=metadata)
        cloud_scs.append(sc)
        cloud_vis.append(vi)
        cloud_pris.append(pri)
        cloud_mious.append(miou)
        cloud_f1s.append(f1)
    noise_scs = {"0.5":[], "1.0":[], "2.0":[], "4.0":[], "8.0":[]}
    noise_vis = {"0.5":[], "1.0":[], "2.0":[], "4.0":[], "8.0":[]}
    noise_pris = {"0.5":[], "1.0":[], "2.0":[], "4.0":[], "8.0":[]}
    noise_mious = {"0.5":[], "1.0":[], "2.0":[], "4.0":[], "8.0":[]}
    noise_f1s = {"0.5":[], "1.0":[], "2.0":[], "4.0":[], "8.0":[]}
    print(f"Evaluating {method_name} on noise dataset (2/3)...")
    for path in noise_segmentation_paths:
        id = path.split("/")[-1].split(".")[0]
        noise_level = path.split("/")[-2]
        if not os.path.exists(path):
            noise_scs[noise_level].append(0)
            noise_vis[noise_level].append(100)
            noise_pris[noise_level].append(0)
            noise_mious[noise_level].append(0)
            noise_f1s[noise_level].append(0)
            continue
        segmentation = Image.open(path)
        if has_been_evaluated(segmentation) and use_cache:
            sc = float(segmentation.text["sc"])
            vi = float(segmentation.text["vi"])
            pri = float(segmentation.text["pri"])
            miou = float(segmentation.text["miou"])
            f1 = float(segmentation.text["f1"])
        else:
            segmentation = np.array(segmentation)
            gt = np.array(Image.open(f"../datasets/noise/gt/{id}.png").convert("L"))
            segmentation = torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0)
            segmentation = torch.nn.functional.interpolate(segmentation, size=gt.shape, mode="nearest").squeeze().numpy()
            sc = calculate_SC(segmentation, gt)
            vi = calculate_VI(segmentation, gt)
            pri = calculate_PRI(segmentation, gt)
            miou = calculate_mIOU(segmentation, gt)
            f1 = calculate_f1(segmentation, gt)
            metadata = PngInfo()
            metadata.add_text("sc", str(sc))
            metadata.add_text("vi", str(vi))
            metadata.add_text("pri", str(pri))
            metadata.add_text("miou", str(miou))
            metadata.add_text("f1", str(f1))
            Image.fromarray(segmentation).save(path, pnginfo=metadata)
        noise_scs[noise_level].append(sc)
        noise_vis[noise_level].append(vi)
        noise_pris[noise_level].append(pri)
        noise_mious[noise_level].append(miou)
        noise_f1s[noise_level].append(f1)
    texture_scs = []
    texture_vis = []
    texture_pris = []
    texture_mious = []
    texture_f1s = []
    print(f"Evaluating {method_name} on texture dataset (3/3)...")
    for path in texture_segmentation_paths:
        id = f'{int(path.split("/")[-1].split(".")[0].split("_")[0]):04d}'
        if not os.path.exists(path):
            texture_scs.append(0)
            texture_vis.append(100)
            texture_pris.append(0)
            texture_mious.append(0)
            texture_f1s.append(0)
            continue
        segmentation = Image.open(path)
        if has_been_evaluated(segmentation) and use_cache:
            sc = float(segmentation.text["sc"])
            vi = float(segmentation.text["vi"])
            pri = float(segmentation.text["pri"])
            miou = float(segmentation.text["miou"])
            f1 = float(segmentation.text["f1"])
        else:
            segmentation = np.array(segmentation)
            gt = np.array(Image.open(f"../datasets/texture/gt/{id}.png").convert("L"))
            segmentation = torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0)
            segmentation = torch.nn.functional.interpolate(segmentation, size=gt.shape, mode="nearest").squeeze().numpy()
            sc = calculate_SC(segmentation, gt)
            vi = calculate_VI(segmentation, gt)
            pri = calculate_PRI(segmentation, gt)
            miou = calculate_mIOU(segmentation, gt)
            f1 = calculate_f1(segmentation, gt)
            metadata = PngInfo()
            metadata.add_text("sc", str(sc))
            metadata.add_text("vi", str(vi))
            metadata.add_text("pri", str(pri))
            metadata.add_text("miou", str(miou))
            metadata.add_text("f1", str(f1))
            Image.fromarray(segmentation).save(path, pnginfo=metadata)
        texture_scs.append(sc)
        texture_vis.append(vi)
        texture_pris.append(pri)
        texture_mious.append(miou)
        texture_f1s.append(f1)
    results_for_method = {
        "clouds": {
            "sc": np.mean(cloud_scs),
            "vi": np.mean(cloud_vis),
            "pri": np.mean(cloud_pris),
            "miou": np.mean(cloud_mious),
            "f1": np.mean(cloud_f1s)
        },
        "noise_0.5": {
            "sc": np.mean(noise_scs["0.5"]),
            "vi": np.mean(noise_vis["0.5"]),
            "pri": np.mean(noise_pris["0.5"]),
            "miou": np.mean(noise_mious["0.5"]),
            "f1": np.mean(noise_f1s["0.5"])
        },
        "noise_1.0": {
            "sc": np.mean(noise_scs["1.0"]),
            "vi": np.mean(noise_vis["1.0"]),
            "pri": np.mean(noise_pris["1.0"]),
            "miou": np.mean(noise_mious["1.0"]),
            "f1": np.mean(noise_f1s["1.0"])
        },
        "noise_2.0": {
            "sc": np.mean(noise_scs["2.0"]),
            "vi": np.mean(noise_vis["2.0"]),
            "pri": np.mean(noise_pris["2.0"]),
            "miou": np.mean(noise_mious["2.0"]),
            "f1": np.mean(noise_f1s["2.0"])
        },
        "noise_4.0": {
            "sc": np.mean(noise_scs["4.0"]),
            "vi": np.mean(noise_vis["4.0"]),
            "pri": np.mean(noise_pris["4.0"]),
            "miou": np.mean(noise_mious["4.0"]),
            "f1": np.mean(noise_f1s["4.0"])
        },
        "noise_8.0": {
            "sc": np.mean(noise_scs["8.0"]),
            "vi": np.mean(noise_vis["8.0"]),
            "pri": np.mean(noise_pris["8.0"]),
            "miou": np.mean(noise_mious["8.0"]),
            "f1": np.mean(noise_f1s["8.0"])
        },
        "texture": {
            "sc": np.mean(texture_scs),
            "vi": np.mean(texture_vis),
            "pri": np.mean(texture_pris),
            "miou": np.mean(texture_mious),
            "f1": np.mean(texture_f1s)
        },
        "all": {
            "sc": np.mean(cloud_scs + noise_scs["2.0"] + texture_scs),
            "vi": np.mean(cloud_vis + noise_vis["2.0"] + texture_vis),
            "pri": np.mean(cloud_pris + noise_pris["2.0"] + texture_pris),
            "miou": np.mean(cloud_mious + noise_mious["2.0"] + texture_mious),
            "f1": np.mean(cloud_f1s + noise_f1s["2.0"] + texture_f1s)
        }
    }
    results[method_name] = results_for_method
    print(f"{method_name} evaluation finished")
    return results_for_method