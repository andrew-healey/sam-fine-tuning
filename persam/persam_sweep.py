import wandb
from persam_f import *
from metrics import semseg_iou

# Define sweep config
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'miou'},
    'parameters': 
    {
        'ft': {'values': FT_EXPERIMENT_NAMES},
        'use_box': {'values': [True,False]},
        'norm': {'values':[True, False]},
        'use_guidance': {'values':[True, False]},
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
    
    args = parser.parse_args()
    return args


def main():
    run = wandb.init()

    # note that we define values from `wandb.config`  
    # instead of defining hard values
    experiment_name =  wandb.config.ft
    use_box = wandb.config.use_box
    should_normalize = wandb.config.norm
    use_guidance = wandb.config.use_guidance

    #
    # Inference
    #
    predictor = load_predictor(sam_type='vit_t')

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
# Start sweep job.
wandb.agent(sweep_id, function=main, count=args.sweep_count)