import argparse 
from ezexr import imread, imwrite
import os 
import skimage
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hdr", type=str, required=True ,help='input directory')
    parser.add_argument("--input_ldr", type=str, required=True ,help='input directory')
    parser.add_argument("--output_ldr", type=str, required=True ,help='input directory')
    parser.add_argument("--idx", type=int, default=0 ,help='Current id of the job (start by index 0)')
    parser.add_argument("--total", type=int, default=1 ,help='Total process avalible')
    return parser

def tonemap(img, gamma=2.2):
    """Apply gamma, then clip between 0 and 1, finally convert to uint8 [0,255]"""
    return (np.clip(np.power(img,1/gamma), 0.0, 1.0)*255).astype('uint8')

def reexpose_hdr(hdrim, percentile=90, max_mapping=0.8, alpha=None):
    """
    :param img: HDR image
    :param percentile:
    :param max_mapping:
    :return:
    """
    r_percentile = np.percentile(hdrim, percentile)
    if alpha==None:
        alpha = max_mapping / (r_percentile + 1e-10)
    return alpha * hdrim, alpha

def process_image(args, filename):
    png_name = filename.replace(".exr", ".png")
    if filename.endswith(".exr"):
        path = os.path.join(args.input_hdr, filename)
        if not os.path.exists(path):
            return None
        image = imread(path)
        image, _  = reexpose_hdr(image)    
        image = tonemap(image)
        skimage.io.imsave(os.path.join(args.output_ldr, png_name), image, check_contrast=False)
    return None

def main():
    args = create_argparser().parse_args()
    os.makedirs(args.output_ldr, exist_ok=True)
    files = sorted(os.listdir(args.input_ldr))[args.idx::args.total]
    fn = partial(process_image, args)
    print("Tonemapping...")
    for filename in tqdm(files):
        fn(filename)
        
            
if __name__ == '__main__':
    main()