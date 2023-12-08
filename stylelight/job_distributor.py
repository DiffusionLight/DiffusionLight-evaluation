import argparse
import os 

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True ,help='output directory') 
    parser.add_argument("--input_dir", type=str, required=True ,help='input directory')
    parser.add_argument("--tasks", type=str, default="mirror,matte_silver,diffuse" ,help='name of the job to do') 
    parser.add_argument("--idx", type=int, default=0 ,help='Current id of the job (start by index 0)')
    parser.add_argument("--total", type=int, default=1 ,help='Total process avalible')
    parser.add_argument("--blender_path", type=str, default="blender" ,help='blender binary')
    parser.add_argument("--batch_size", type=int, default=10, help='image per batch')
    return parser

def main():
    args = create_argparser().parse_args()
    files = os.listdir(args.input_dir)
    total_files  = len(files)
    # https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    files_for_this_thread = (total_files + args.total - 1) // args.total
    real_begin = files_for_this_thread * args.idx
    real_ends = files_for_this_thread * (args.idx+1)
    loop_counter = 0
    job_names = [f.strip() for f in args.tasks.split(",")]
    while (loop_counter * args.batch_size) < (real_ends - real_begin):
        begin = loop_counter * args.batch_size + real_begin
        finish = min(begin + args.batch_size, real_ends)
        for job_name in job_names:
            if job_name == "diffuse":
                cmd = f'{args.blender_path} --background --python diffuse.py -- 50 5 "{args.input_dir}" "{args.output_dir}" {begin} {finish}'
            elif job_name == "matte_silver":
                cmd = f'{args.blender_path} --background --python matte_silver.py -- 50 5 "{args.input_dir}" "{args.output_dir}" {begin} {finish}'
            elif job_name == "mirror":
                cmd = f'{args.blender_path} --background --python mirror.py -- 50 5 {args.input_dir} {args.output_dir} {begin} {finish}'
            else:
                raise NotImplementedError()
            print(cmd)
            os.system(cmd)
        loop_counter += 1
           
    
if __name__ == "__main__":
    main()