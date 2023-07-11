import wandb
from persam_f import *
from metrics import semseg_iou

# Define sweep config
sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'miou'},
    'parameters': 
    {
        'ft': {'values': FT_EXPERIMENT_NAMES},
        'use_box': {'values': BOX_EXPERIMENT_VALUES},
        'norm': {'values':NORM_EXPERIMENT_VALUES},
        'use_guidance': {'values':GUIDANCE_EXPERIMENT_VALUES},
     }
}

# Initialize sweep by passing in config. 
# (Optional) Provide a name of the project.
sweep_id = wandb.sweep(
  sweep=sweep_configuration, 
  project='persam-sweep'
  )

import argparse
def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref_img', type=str, default='./data/Images/*/00.jpg')
    parser.add_argument('--ref_mask', type=str, default='./data/Annotations/*/00.png')
    parser.add_argument('--img_dir', type=str, default='./data/Images/*')
    parser.add_argument('--gt_dir', type=str, default='./data/Annotations/*')
    parser.add_argument('--out_dir', type=str, default='output')

    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--sweep_count', type=int, default=50)

    parser.add_argument('--run_once', action='store_true')

    parser.add_argument('--experiment', type=str, default='single')

    # TODO: rename this to "mean"
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--no-norm',dest='norm',action='store_false')
    parser.set_defaults(norm=True)

    parser.add_argument('--box', action='store_true')
    parser.add_argument('--no-box',dest='box',action='store_false')
    parser.set_defaults(box=True)
    
    parser.add_argument('--guidance', action='store_true')
    parser.add_argument('--no-guidance',dest='guidance',action='store_false')
    parser.set_defaults(guidance=True)
    
    args = parser.parse_args()
    return args


def main():

    if run_once:
        experiment_name = args.experiment
        should_normalize = args.norm
        use_box = args.use_box
        use_guidance = args.use_guidance
        sam_type = args.sam_type
    else:
        run = wandb.init()
        experiment_name =  wandb.config.ft
        use_box = wandb.config.use_box
        should_normalize = wandb.config.norm
        use_guidance = wandb.config.use_guidance
        sam_type = 'vit_t'

    #
    # Inference
    #
    predictor = load_predictor(sam_type=sam_type)

    rmrf(args.out_dir)
    mkdirp(args.out_dir)

    for ref_img_path,ref_mask_path,test_img_dir,output_dir in load_dirs(args.ref_img,args.ref_mask,args.img_dir,args.out_dir):
        persam_f(predictor,ref_img_path,ref_mask_path,test_img_dir,output_dir,experiment_name,should_normalize,use_box,use_guidance)
    
    print("done with inference")

    #
    # Evaluation
    #
    running_iou = 0
    num_dirs = 0

    gt_dirs = glob.glob(args.gt_dir)

    for output_dir in glob.glob(os.path.join(args.out_dir,"*")):
        gt_dir = [d for d in gt_dirs if os.path.basename(d) == os.path.basename(output_dir)][0]

        miou = semseg_iou(gt_dir,output_dir)
        g = os.path.basename(gt_dir)
        num_dirs+=1
        running_iou+=miou
    
    miou = running_iou / num_dirs
    print("miou: ",miou)
    
    wandb.log({
        'miou': miou,
    })

args = get_arguments()
run_once = args.run_once

if run_once:
    main()
    exit()
else:
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=args.sweep_count)