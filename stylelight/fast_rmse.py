import numpy as np
import hdrio
import os
import random
import argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--fake', help='test data. eg:test5_tone/mirror')
parser.add_argument('--real', help='mirror / matte_silver / diffuse')

args = parser.parse_args()

use_debug = False
hdr_clip = False
all_pixels = False

def rmse(pred, gt, mask=None):
    h = pred.shape[0]
    w = pred.shape[1]
    if mask is None:
        if all_pixels:
            mask = np.ones((h, w))
        else:
            mask = np.zeros((h, w))
            mask[pred[:,:,3]>0] = 1
    


    num_pixel = mask.sum()
    if use_debug:
        pred = np.ones_like(pred)*1e-5
    error_ = (pred[:,:,:3] - gt[:,:,:3])**2
    mask_error = np.multiply(error_, np.repeat(mask[:,:,np.newaxis],3,axis=2))
    mask_error = mask_error.sum()/num_pixel
    res = np.sqrt(mask_error)

    return res


def si_rmse(pred, gt, mask=None):
    h = pred.shape[0]
    w = pred.shape[1]
    if mask is None:
        if all_pixels:
            mask = np.ones((h, w))
        else:
            mask = np.zeros((h, w))
            mask[pred[:,:,3]>0] = 1
   
    num_pixel = mask.sum()

    if use_debug:
        pred = np.ones_like(pred)*1e-5

    x_hat_square_ = pred[:, :, :3]**2

    x_square_ = gt[:,:,:3] ** 2
    x_x_hat_ = -(2 * pred[:,:,:3] * gt[:,:,:3])
    x_hat_square_ = np.multiply(x_hat_square_, np.repeat(mask[:, :, np.newaxis], 3, axis=2))
    x_square_= np.multiply(x_square_, np.repeat(mask[:, :, np.newaxis], 3, axis=2))
    x_x_hat_ = np.multiply(x_x_hat_, np.repeat(mask[:, :, np.newaxis], 3, axis=2))

    
    a = x_hat_square_.sum()
    b = x_x_hat_.sum()
    c = x_square_.sum()
   # print(a,b,c)
   #  if 0 not in a:
    mask_error = (4*a*c-b**2)/(4*a)

    #alpha = -b/(2*a)
    #print(alpha)
    #re = rmse(alpha*pred, gt)
    #print(mask_error)

    mask_error = mask_error / num_pixel
    res = np.sqrt(mask_error)

    #print(re,res)

    return res

# fast_rmse.py:71: RuntimeWarning: invalid value encountered in true_divide                                                                                                                  
#   cos = np.sum(gt[:,:,:3]*pred[:,:,:3], axis=2) / (np.linalg.norm(gt[:,:,:3],axis=2) * np.linalg.norm(pred[:,:,:3],axis=2)) 

def angular(pred, gt, mask = None):
    h = pred.shape[0]
    w = pred.shape[1]
    if mask is None:
        mask = np.zeros((h, w))
        mask[pred[:,:,3]>0] = 1
    num_pixel = mask.sum()

    cos = np.sum(gt[:,:,:3]*pred[:,:,:3], axis=2) / (np.linalg.norm(gt[:,:,:3],axis=2) * np.linalg.norm(pred[:,:,:3],axis=2))
    cos = np.clip(cos, 0, 1)
    angular=np.nan_to_num(np.multiply(np.arccos(cos), mask))

    res=np.degrees(angular.sum()/num_pixel)

    return res

def get_mask(img):
    h,w = img.shape[:2]
    mask = np.zeros((h, w))
    mask[img[:,:,3]>0] = 1
    return mask

def normalize(x):
    # Taken form ExpandNet
    return (x - np.percentile(x,0.01)) / (np.percentile(x,99.9) - np.percentile(x,0.01))

#fake_path = "/home/ynyang/Downloads/blender-cli-rendering-master/emlight_tone_diffuse/"
#real_path = '/home/ynyang/Downloads/blender-cli-rendering-master/ground_truth/test_tone_diffuse/'
#real_path_2='/home/ynyang/Downloads/blender-cli-rendering-master/ground_truth/s_new/'

"""
tonemap = True
if tonemap:
    fake_path = './data/render_results/' + args.fake + '/'
    #real_path = './data/ground_truth/test_tone_'+args.real+'/'
    #real_path = './data/ground_truth_ours_neg0.6_60degree/test_tone_'+args.real+'/'
    real_path = './data/ground_truth_ours_neg0.6_60degree_HR/test_tone_'+args.real+'/'
    # real_path = './data/ground_truth_flip/test_tone_'+args.real+'/'
else:
    fake_path = './data/render_results/' + args.fake + '/'
    #real_path = './data/ground_truth/test'+args.real+'/'
    #real_path = './data/ground_truth_ours_neg0.6_60degree/test'+args.real+'/'
    real_path = './data/ground_truth_ours_neg0.6_60degree_HR/test'+args.real+'/'
    # real_path = './data/ground_truth_flip/test'+args.real+'/'
"""
fake_path = args.fake 
real_path = args.real

print('args.fake:',args.fake)
fake_imgs = os.listdir(fake_path)
fake_imgs.sort()

real_imgs = os.listdir(real_path)
#real_imgs.sort()
random.shuffle(real_imgs)

csv_content = "name, rmse, si_rmse, angular_error, norm_rmse\n"

r=0
si_r=0
ang = 0
cnt=0
#PURE: Add norm_rmse
norm_r = 0

for i, fake in enumerate(tqdm(fake_imgs)):
    pred = hdrio.imread(os.path.join(fake_path,fake))
    
    #PURE: add mask
    mask = get_mask(pred.copy())
    
    if hdr_clip:
        pred = np.clip(pred, 0, 0.5)
        # pred = np.ones_like(pred)*0.1
        # print('lwq')
    # pred = hdrio.imread("/home/ynyang/Downloads/blender-cli-rendering-master/gardner_mirror/black.exr")
    gt_new = True #False #True #False
    if gt_new:
        real = fake
    else:
        real = fake.replace("_fake_image","")

        # real = real[:8]+'_fake_image.exr'
        real = real[:8]+'.exr'

    # print(fake)
    # print(real)
    gt = hdrio.imread(real_path+real)


    # gt2 = hdrio.imread(real_path_2+real)
    #gt = hdrio.imread(real_path+real_imgs[i])
    r1=rmse(pred,gt)
    #r2=rmse(pred,gt2)
    sr1=si_rmse(pred,gt)
    # sr2 = si_rmse(pred, gt2)
    ang1 = angular(pred,gt)
    # print("=======> i:",i)
    # if np.isnan(ang1):        
    #     raise Exception('nan ang1')
    # if np.isnan(r1):
    #     raise Exception('nan r1')
    # if np.isnan(sr1):
    #     raise Exception('nan sr1')
    
    #PURE: add normalize_score
    norm_pred = normalize(pred[...,:3])
    norm_gt = normalize(gt[...,:3])
    
    norm_r1 = rmse(norm_pred, norm_gt, mask)
    
    csv_content = csv_content + f"{fake}, {r1}, {sr1}, {ang1}, {norm_r1}\n"
    norm_r += norm_r1
    r += r1
    si_r += sr1
    ang+=ang1
    cnt += 1
#print(cnt)

# if args.real == 'mirror':
#     print("Mirror")
# elif args.real == 'silver':
#     print("Matte Silver")
# elif args.real == 'diffuse':
#     print("Diffuse")

# print(r/cnt, si_r/cnt, ang/cnt)
# print(r/cnt, si_r/cnt)

name = args.fake.split('/')[-2]
content = f"name, rmse, si_rmse, angular_error, norm_rmse\n{name}, {r/cnt}, {si_r/cnt}, {ang/cnt}, {norm_r/cnt}"
print(name)
print(content)

## write file back for later verufy 
with open(fake_path+f'/../{name}.csv', 'w') as f:
    f.write(content)

with open(fake_path+f'/../{name}_scenes.csv', 'w') as f:
    f.write(csv_content)