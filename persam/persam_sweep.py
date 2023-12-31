import wandb
from persam_f import *
from metrics import semseg_iou

import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ref_img", type=str, default="./data/Images/*/00.jpg")
    parser.add_argument("--ref_mask", type=str, default="./data/Annotations/*/00.png")
    parser.add_argument("--img_dir", type=str, default="./data/Images/*")
    parser.add_argument("--gt_dir", type=str, default="./data/Annotations/*")
    parser.add_argument("--out_dir", type=str, default="output")

    parser.add_argument("--sam_type", type=str, default="vit_h")
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--sweep_count", type=int, default=50)
    parser.add_argument("--search_method", type=str, default="random")
    parser.add_argument("--run_once", action="store_true")
    parser.add_argument("--cache_encoder", action="store_true")

    parser.add_argument("--experiment", type=str, default="single")

    # TODO: rename this to "mean"
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--no-norm", dest="norm", action="store_false")
    parser.set_defaults(norm=True)

    parser.add_argument("--box", action="store_true")
    parser.add_argument("--no-box", dest="box", action="store_false")
    parser.set_defaults(box=True)

    parser.add_argument("--attn", action="store_true")
    parser.add_argument("--no-attn", dest="attn", action="store_false")
    parser.set_defaults(attn=True)

    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--no-embed", dest="embed", action="store_false")
    parser.set_defaults(embed=True)

    parser.add_argument("--neg", action="store_true")
    parser.add_argument("--no-neg", dest="neg", action="store_false")
    parser.set_defaults(neg=True)

    args = parser.parse_args()
    return args

def run_with_id(fn, id):
    def wrapper():
        fn(id)
    return wrapper

def main(id:int=None):
    print("id:", id)
    if run_once:
        experiment_name = args.experiment
        should_normalize = args.norm
        use_box = args.box
        use_attn = args.attn
        use_embed = args.embed
        include_neg = args.neg
        sam_type = args.sam_type
    else:
        run = wandb.init()
        experiment_name = wandb.config.ft
        use_box = wandb.config.use_box
        should_normalize = wandb.config.norm
        use_attn = wandb.config.attn
        use_embed = wandb.config.embed
        include_neg = wandb.config.neg
        sam_type = args.sam_type

    #
    # Inference
    #

    cache_encoder = args.cache_encoder

    # Only actually *load* MobileSAM, but use cached SAM for inference
    if cache_encoder:
        print("caching encoder")
        predictor = load_predictor(sam_type="vit_t")
        predictor.model.sam_type = sam_type
    else:
        predictor = load_predictor(sam_type=sam_type)
    
    if id is not None:
        out_dir = os.path.join(args.out_dir, str(id))
    else:
        out_dir = args.out_dir

    rmrf(out_dir)
    mkdirp(out_dir)

    for ref_img_path, ref_mask_path, test_img_dir, output_dir in load_dirs(
        args.ref_img, args.ref_mask, args.img_dir, out_dir
    ):
        persam_f(
            predictor,
            ref_img_path,
            ref_mask_path,
            test_img_dir,
            output_dir,
            experiment_name,
            should_normalize,
            use_box,
            use_attn,
            use_embed,
            include_neg,
        )

    print("done with inference")

    #
    # Evaluation
    #
    running_iou = 0
    num_dirs = 0

    gt_dirs = glob.glob(args.gt_dir)

    for output_dir in glob.glob(os.path.join(out_dir, "*")):
        gt_dir = [
            d for d in gt_dirs if os.path.basename(d) == os.path.basename(output_dir)
        ][0]

        miou = semseg_iou(gt_dir, output_dir)
        g = os.path.basename(gt_dir)
        num_dirs += 1
        running_iou += miou

    miou = running_iou / num_dirs
    print("miou: ", miou)

    if not run_once:
        wandb.log(
            {
                "miou": miou,
                "id": id,
            }
        )

        rmrf(out_dir)


args = get_arguments()
run_once = args.run_once

if run_once:
    main()
    exit()

# Define sweep config
sweep_configuration = {
    "method": args.search_method,
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "miou"},
    "parameters": {
        "ft": {"values": FT_EXPERIMENT_NAMES},
        "use_box": {"values": BOX_EXPERIMENT_VALUES},
        "norm": {"values": NORM_EXPERIMENT_VALUES},
        "attn": {"values": ATTN_EXPERIMENT_VALUES},
        "embed": {"values": EMBED_EXPERIMENT_VALUES},
        "neg": {"values": NEG_EXPERIMENT_VALUES},
    },
}

# Initialize sweep by passing in config.
# (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="persam-sweep")


from math import ceil
num_agents = args.num_agents
runs_per_agent = ceil(args.sweep_count / num_agents)
import multiprocessing

def run_agent(id):
    my_main = run_with_id(main, id)
    wandb.agent(sweep_id, function=my_main, count=runs_per_agent)

with multiprocessing.Pool(num_agents) as p:
    p.map(run_agent, range(num_agents))
