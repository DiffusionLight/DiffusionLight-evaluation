from ezexr import imread
import os
import numpy as np
from tqdm import tqdm
from math import log10
import pandas as pd
import argparse

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True ,help='input directory (directory to compute the score)')
    parser.add_argument("--output_dir", type=str, required=True ,help='output directory') 
    parser.add_argument("--gt_dir", type=str, required=True ,help='ground truth directory') 
    return parser

args = create_argparser().parse_args()

OUTPUT_FOLDER = args.output_dir
OUR_DIR = args.input_dir
GT_DIR = args.gt_dir


def wrmse(gt, est, mask):
    if mask is None:
        gt = gt.flatten()
        est = est.flatten()
    else:
        gt = gt[mask].flatten()
        est = est[mask].flatten()
    error = np.sqrt(np.mean(np.power(gt - est, 2)))

    return error

def si_wrmse(gt, est, mask):
    if mask is None:
        gt_c = gt.flatten()
        est_c = est.flatten()
    else:
        gt_c = gt[mask].flatten()
        est_c = est[mask].flatten()
    alpha = (np.dot(np.transpose(gt_c), est_c)) / (np.dot(np.transpose(est_c), est_c))
    error = wrmse(gt, est * alpha, mask)

    return error

def angular_error(gt_render, pred_render, mask=None):
    # The error need to be computed with the normalized rgb image.
    # Normalized RGB is r = R / (R+G+B), g = G / (R+G+B), b = B / (R+G+B)
    # The angular distance is the distance between pixel 1 and pixel 2.
    # It's computed with cos^-1(p1·p2 / ||p1||*||p2||)
    gt_norm = np.empty((gt_render.shape))
    pred_norm = np.empty(pred_render.shape)

    for i in range(3):
        gt_norm[:,:,i] = gt_render[:,:,i] / np.sum(gt_render, axis=2, keepdims=True)[:,:,0]
        pred_norm[:,:,i] = pred_render[:,:,i] / (np.sum(pred_render, axis=2, keepdims=True)[:,:,0] + 1e-8)

    angular_error_arr = np.arccos( np.sum(gt_norm*pred_norm, axis=2, keepdims=True)[:,:,0] / 
        ((np.sqrt(np.sum(gt_norm*gt_norm, axis=2, keepdims=True)[:,:,0])*np.sqrt(np.sum(pred_norm*pred_norm, axis=2, keepdims=True)[:,:,0]))) )

    if mask is not None:
        angular_error_arr = angular_error_arr[mask[:,:,0]]
    else:
        angular_error_arr = angular_error_arr.flatten()
    angular_error_arr = angular_error_arr[~np.isnan(angular_error_arr)]
    mean = np.mean(angular_error_arr)
    # convert to degree
    mean = mean * 180 / np.pi
    return mean

def psnr(original, compressed):
    mse = wrmse(original, compressed, None)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = max(original.max(), compressed.max())
    psnr = 20 * log10(max_pixel / mse)
    return psnr

def remove_nan_and_sort(x):
    x = np.array(x)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    return x

results_dataset_roots = [
                         #GT_DIR,
                         OUR_DIR,
                        ]

dataset_names = [
                #'gt',
                'ours', 
                ]

mse = {}
si_mse = {}
angular = {}
psnr_error = {}
csv_info = []
for key in dataset_names:
    mse[key] = []
    si_mse[key] = []
    angular[key] = []
    psnr_error[key] = []

gt_dataset_root = GT_DIR
output_folder = OUTPUT_FOLDER
print("------------------------------------")
print("Ground truth dataset root: {}".format(gt_dataset_root))
print("Output folder: {}".format(output_folder))
print("------------------------------------")

gt_dataset_files = sorted([os.path.join(gt_dataset_root, f) for f in os.listdir(gt_dataset_root) if f.endswith('.exr')])

# # filter out the files that are not in the results_dataset_roots
for results_dataset_root in results_dataset_roots:
    print('analyzing results in ' + results_dataset_root)
    gt_dataset_files = [f for f in gt_dataset_files if os.path.exists(f.replace(gt_dataset_root, results_dataset_root))]
    print('size of gt_dataset_files: {}'.format(len(gt_dataset_files)))

for gt_dataset_file in tqdm(gt_dataset_files):
    gt_dataset_img_exr = imread(gt_dataset_file)[:, :, :3].astype(np.float32)
    
    for results_dataset_root, dataset_name in zip(results_dataset_roots, dataset_names):
        result_dataset_file = gt_dataset_file.replace(gt_dataset_root, results_dataset_root)
        result_dataset_img_exr = imread(result_dataset_file)[:, :, :3].astype(np.float32)
        
        mse_result = wrmse(gt_dataset_img_exr, result_dataset_img_exr, None) 
        scale_invariant_mse_result = si_wrmse(gt_dataset_img_exr, result_dataset_img_exr, None)
        angular_result = angular_error(gt_dataset_img_exr, result_dataset_img_exr, None)
        psnr_result = psnr(gt_dataset_img_exr, result_dataset_img_exr)
        mse[dataset_name].append(mse_result)
        si_mse[dataset_name].append(scale_invariant_mse_result)
        angular[dataset_name].append(angular_result)
        psnr_error[dataset_name].append(psnr_result)
        si_mse[dataset_name].append(scale_invariant_mse_result)
        csv_info.append([os.path.basename(gt_dataset_file), mse_result, scale_invariant_mse_result, angular_result, psnr_result])

# create output folder output_folder, 'metrics'
os.system('mkdir -p ' + output_folder + '/editableindoor_metrics')

mse = {key: remove_nan_and_sort(mse[key]) for key in mse}

mse_ours = np.percentile(mse['ours'], q=50)
si_mse_ours = np.percentile(si_mse['ours'], q=50)
angular_ours = np.percentile(angular['ours'], q=50)
psnr_ours = np.percentile(psnr_error['ours'], q=50)
print('mse_ours: {}'.format(mse_ours))
print('si_mse_ours: {}'.format(si_mse_ours))
print('angular_ours: {}'.format(angular_ours))
print('psnr_ours: {}'.format(psnr_ours))
df = pd.DataFrame(csv_info, columns=['Name', ',mse', 'si_mse', 'angular', 'psnr'])
df.to_csv('{}/scenes.csv'.format(output_folder), index=False)

print('{}/scenes.csv'.format(output_folder))
print('{}/metrics.csv'.format(output_folder))
with open('{}/metrics.csv'.format(output_folder), 'w') as f:
    f.write('mse, si_mse, angular, psnr\n')
    f.write(f'{mse_ours},{si_mse_ours},{angular_ours},{psnr_ours}\n')