# DiffusionLight-evaluation: an evaluation code using in DiffusionLight
### [Project Page](https://diffusionlight.github.io/) | [Main Repository](https://github.com/DIffusionLight/DiffusionLight)

We provide a **slightly** modified version of the evaluation code from [StyleLight](https://style-light.github.io/) and [Editable Indoor LightEstimation](https://arxiv.org/abs/2211.03928). While we adapted the code to better suit our needs, the underlying evaluation method remains the same. You may use either this repository or the original code; both should produce identical score outputs.

This directory contains an example predicted output and an example ground truth in the `example` directory.


## Table of contents
-----
  * [Poly Haven and Laval indoor](#Poly-Haven-and-Laval-Indoor)
  * [Array of spheres](#Array-of-spheres)
  * [Installation](#Installation)
  * [Citation](#Citation)
------

### Poly Haven and Laval indoor

```shell
cd stylelight
python envmap2resize.py --input_dir ../example/hdr/stylelight/ --output_dir ../output/stylelight/envmap_128x256
python tonemap.py --testdata ../output/stylelight/envmap_128x256 --out_dir ../output/stylelight/envmap_toned
python job_distributor.py --input_dir ../output/stylelight/envmap_toned --output_dir ../output/stylelight/rendered
sh test_rmse.sh ../output/stylelight/rendered ../example/ground_truth/stylelight
```

The score will be printed to the terminal and saved to a CSV file at .  `../output/stylelight/rendered`. 

Even though both DiffusionLight and StyleLight are outputs at 256x512, we still run `envmap2resize.py` to resize the environment map to 128x256 to match the StyleLight evaluation method described in the implementation details of the StyleLight paper.

### Array of sphere

```shell
cd editableindoor
python job_distributor.py --input_dir ../example/hdr/editableindoor/ --output_dir ../output/editableindoor/hdr --output_ldr ../output/editableindoor/ldr 
python res1_table.py --input_dir ../output/editableindoor/hdr --output_dir ../output/editableindoor --gt_dir ../example/ground_truth/editableindoor
```

The score will be printed to the terminal and saved to a CSV file at   `../output/editableindoor`

## Installation
### 1. Python environment setup 

Please follow the instructions in our main repository's [installation guide](https://github.com/DiffusionLight/DiffusionLight#Installation)

### 2. Install Blender

Install Blender on your machine. At the time of writing this paper, we were using Blender version 3.6.5. However, any version of Blender newer than 3.0 should work.

## Citation

```
@inproceedings{Phongthawee2023DiffusionLight,
    author = {Phongthawee, Pakkapon and Chinchuthakun, Worameth and Sinsunthithet, Nontaphat and Raj, Amit and Jampani, Varun and Khungurn, Pramook and Suwajanakorn, Supasorn},
    title = {DiffusionLight: Light Probes for Free by Painting a Chrome Ball},
    booktitle = {ArXiv},
    year = {2023},
}
```

## Visit us 🦉
[![Vision & Learning Laboratory](https://i.imgur.com/hQhkKhG.png)](https://vistec.ist/vision) [![VISTEC - Vidyasirimedhi Institute of Science and Technology](https://i.imgur.com/4wh8HQd.png)](https://vistec.ist/)