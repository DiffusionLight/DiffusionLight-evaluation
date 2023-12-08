# PURE
# down scale environment map to 128x256 as per mention in the paper 

import os 
from tqdm.auto import tqdm

from envmap import EnvironmentMap
from hdrio import imsave
from multiprocessing import Pool
import argparse
from functools import partial


def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="" ,help='the input directory that contain EXR file') 
    parser.add_argument("--output_dir", type=str, default="" ,help='Output directory to save the resized file') 
    parser.add_argument("--width", type=int, default=256 ,help='environment map width')
    parser.add_argument("--height", type=int, default=128 ,help='environment map width')
    return parser

def process_image(args, filename):
    ori_path = os.path.join(args.input_dir, filename)
    # read environment map
    e = EnvironmentMap(ori_path, 'latlong')
    
    # To match Stylelight code, this one will use skylibs to resize (but skylibs seem to have aliasing)
    e.resize((128, args.width)) # resize to given size
    imsave(os.path.join(args.output_dir,filename), e.data)

def main():
    args = create_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(os.listdir(args.input_dir))
    fn = partial(process_image, args)
    with Pool(8) as p:
        r = list(tqdm(p.imap(fn, files), total=len(files)))
        
    # for filename in tqdm(files):
    #     ori_path = os.path.join(args.input_dir, filename)
    #     # read environment map
    #     e = EnvironmentMap(ori_path, 'latlong')
    #     e.resize((128, 256)) # resize to 128x256
    #     imsave(os.path.join(args.output_dir,filename), e.data)
    
    
    
if __name__ == "__main__":
    main()