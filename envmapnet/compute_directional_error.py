# compute_directional_error.py

import argparse
from ParametricLights import extract_lights_from_equirectangular_image, calculate_angular_error_metric
from hdrio import imread as exr_read
import os
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help='directory that contain the image') #dataset name or directory 
    parser.add_argument("--output_dir", type=str, required=True, help='directory that will output computed content') #dataset name or directory 
    parser.add_argument("--gt_dir", type=str, required=True, help='ground truth directory') #dataset name or directory 
    return parser


def compute_score(args, filename):
    # load gt image 
    gt_path = os.path.join(args.gt_dir, filename)
    gt_image = exr_read(gt_path)

    # laod input image
    input_path = os.path.join(args.input_dir, filename)
    input_image = exr_read(input_path)
    
    # extract light
    gt_lights = extract_lights_from_equirectangular_image(gt_image)
    input_lights = extract_lights_from_equirectangular_image(input_image)
    
    # compute score
    score = calculate_angular_error_metric(gt_lights, input_lights)
    return [filename,score]

def main():
    args = create_argparser().parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    files = os.listdir(args.gt_dir)
    files = sorted([file for file in files if file.endswith(".exr")])
    scores = []

    with Pool(8) as p:
        scores = list(tqdm(p.imap(partial(compute_score, args), files), total=len(files)))

        # create csv file
        with open(os.path.join(args.output_dir, "directional_score.csv"), "w") as f:
            f.write("filename,score\n")
            for score in scores:
                f.write(f"{score[0]},{score[1]}\n")

        # compute average score
        total_score = 0
        count_score = 0
        for score in scores:
            if score[1] == -1:
                # filterout the image that has no light
                continue
            total_score += score[1]
            count_score += 1
        avg_score = total_score / count_score
        avg_score = avg_score  * 180 / np.pi
        print(f"average score: {avg_score}")
        with open(os.path.join(args.output_dir, "directional_score_average.txt"), "w") as f:
            f.write(str(avg_score))

if __name__ == "__main__":
    main()