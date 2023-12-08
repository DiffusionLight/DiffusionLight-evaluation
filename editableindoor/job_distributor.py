import argparse
import os 

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True ,help='output directory') 
    parser.add_argument("--output_ldr", type=str, required=True ,help='output directory') 
    parser.add_argument("--input_dir", type=str, required=True ,help='input directory')
    parser.add_argument("--idx", type=int, default=0 ,help='Current id of the job (start by index 0)')
    parser.add_argument("--total", type=int, default=1 ,help='Total process avalible')
    parser.add_argument("--blender_path", type=str, default="blender" ,help='blender binary')
    return parser

def main():
    args = create_argparser().parse_args()
    files = sorted(os.listdir(args.input_dir))
    total_files  = len(files)
    for filename in files[args.idx::args.total]:
        if filename.endswith(".exr"):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            if not os.path.exists(output_path):
                os.system(f"{args.blender_path} -b -P blender_script.py -- {input_path} {output_path}")
    os.system(f"python convert2ldr.py --input_ldr {args.input_dir} --input_hdr {args.output_dir} --output_ldr {args.output_ldr} --idx {args.idx} --total {args.total}")
            
    
           
    
if __name__ == "__main__":
    main()