from common import *
from load import *

NORM_EXPERIMENT_VALUES=[True,False]

FT_EXPERIMENT_NAMES=[
    "sam_embedding", # cosine similarity of SAM embeddings
    "clip_embedding", # cosine similarity of CLIP embeddings
    "area", # closest log mask area
    "bbox_area", # closest log bbox area
    "perimeter", # closest log bbox perimeter
    "best_idx", # Use the best idx from the ref inference
    "best_idx_iou", # Use the best idx from the ref inference
    "linear_combo", # Linear combination of SAM embeddings
    "single", # Recreates PerSAM
    "sim", # Max avg similarity
    "max_score", # Mask with highest score
]

def persam_f(predictor:SamPredictor, ref_img_path:str,ref_mask_path:str,test_img_dir:str,output_dir:str, experiment_name:str,should_normalize:bool):

    if experiment_name not in FT_EXPERIMENT_NAMES:
        raise ValueError(f"Invalid experiment name {experiment_name}")

    print("Loading reference image...")
    ref_img,ref_mask = load_image(predictor,ref_img_path,ref_mask_path)

    target_feat,target_embedding,feat_dims = get_mask_embed(predictor,ref_mask,should_normalize)

    mkdirp(output_dir)

    raw_img_pairs = load_images_in_dir(test_img_dir)
    test_img_pairs = [img_pair for img_pair in raw_img_pairs if img_pair[1] != ref_img_path]
    ref_img_pair = (f"REF_{os.path.basename(ref_img_path)}",ref_img_path)
    img_pairs = [ref_img_pair] + test_img_pairs

    for test_img_name,test_img_path in img_pairs:
        is_ref = test_img_path == ref_img_path

        print(f"Processing {test_img_name}...")
        if not is_ref:
            load_image(predictor,test_img_path)

        sim_map = get_sim_map(predictor,target_feat)
        attn_sim = sim_map_to_attn(sim_map)
        points = sim_map_to_points(sim_map)

        kwargs = points_to_kwargs(points)
        target_guidance = {} if is_ref else {
            "attn_sim":attn_sim,  # Target-guided Attention
            "target_embedding":target_embedding  # Target-semantic Prompting
        }

        # Experiments!
        if is_ref: 
            if experiment_name in ["linear_combo","best_idx","best_idx_iou"]:
                logit_weights = get_logit_weights(predictor,ref_mask,experiment_name,target_guidance,**kwargs)
            elif experiment_name == "clip_embedding":
                raise NotImplementedError()
            elif experiment_name == "area":
                ref_area = torch.sum(ref_mask>0)
            elif experiment_name in ["bbox_area","perimeter"]:
                gt_mask = F.interpolate(ref_mask, size=feat_dims, mode="bilinear")
                gt_mask = gt_mask.squeeze()[0]
                box = mask_to_box(gt_mask.cpu().detach().numpy())[0]
                width = box[2] - box[0]
                height = box[3] - box[1]
                ref_area = width * height
                ref_perimeter = 2 * (width + height)
        else:
            mask_picking_data = None
            if experiment_name in ["linear_combo","best_idx","best_idx_iou"]:
                mask_picking_data = logit_weights
            elif experiment_name in ["area","bbox_area"]:
                mask_picking_data = ref_area
            elif experiment_name == "perimeter":
                mask_picking_data = ref_perimeter
            elif experiment_name == "clip_embedding":
                raise NotImplementedError()
            elif experiment_name == "sam_embedding":
                mask_picking_data = (target_embedding,should_normalize)
            elif experiment_name == "sim":
                mask_picking_data = sim_map

            mask = predict_mask_refined(predictor,target_guidance,experiment_name,mask_picking_data,**kwargs)

            mask_path = os.path.join(output_dir,test_img_name+".png")
            save_mask(mask,mask_path)
            print("Saved mask to",mask_path)

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as TVF

lr = 1e-3
train_epoch = 1000
log_epoch = 200

def get_logit_weights(predictor:SamPredictor,ref_mask:torch.Tensor,experiment_name:str,target_guidance:Dict[str,torch.Tensor],**kwargs)->torch.Tensor:
    kwargs = {
        **kwargs,
        "high_res": True,
    }

    # Simulated first-step prediction
    masks, scores, logits, original_logits_high = predictor.predict(
        **kwargs,
        **target_guidance,
        multimask_output=True
    )

    resolution = [256, 256]
    original_logits_high = TVF.resize(original_logits_high,resolution)
    original_logits_high = original_logits_high.flatten(1)

    gt_mask = torch.tensor(ref_mask)[None,:, :, 0] > 0
    gt_mask = TVF.resize(gt_mask.float(), resolution)
    gt_mask = gt_mask.flatten(1).cuda()

    masks = TVF.resize(torch.from_numpy(masks), resolution).flatten(1).cuda()

    # Figure out which logit/mask combination to use.

    # Experiment 1. Find the mask with the highest IoU (or maybe loss) with the target mask.
    # Experiment 2. Run the usual gradient descent on linear weightings of the logits.

    if experiment_name == "linear_combo":
        print('======> Start Training')
        # Learnable mask weights
        mask_weights = Mask_Weights().cuda()
        mask_weights.train()
        
        optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch)

        for train_idx in range(train_epoch):
            # Weighted sum three-scale masks
            weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            curr_logits_high = original_logits_high * weights
            curr_logits_high = curr_logits_high.sum(0).unsqueeze(0)

            dice_loss = calculate_dice_loss(curr_logits_high, gt_mask)
            focal_loss = calculate_sigmoid_focal_loss(curr_logits_high, gt_mask)
            loss = dice_loss + focal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if train_idx % log_epoch == 0:
                print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))
                current_lr = scheduler.get_last_lr()[0]
                print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))


        mask_weights.eval()
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)

        return weights

    best_idx = -1

    if experiment_name == "best_idx":
        dice_losses = [calculate_dice_loss(original_logits_high[None,idx],gt_mask) for idx in range(3)]
        focal_losses = [calculate_sigmoid_focal_loss(original_logits_high[None,idx],gt_mask) for idx in range(3)]
        losses = [dice_losses[idx] + focal_losses[idx] for idx in range(3)]
        best_idx = torch.argmin(torch.stack(losses))

    elif experiment_name == "best_idx_iou":
        intersections = [torch.logical_and(masks[idx],gt_mask).sum() for idx in range(3)]
        unions = [torch.logical_or(masks[idx],gt_mask).sum() for idx in range(3)]
        ious = [intersections[idx] / (unions[idx] + 1e-10) for idx in range(3)]
        best_idx = torch.argmax(torch.stack(ious))

    assert best_idx != -1, "Invalid experiment name"

    return F.one_hot(best_idx,3).float().cuda()

class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def calculate_dice_loss(inputs, targets, num_masks = 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


import argparse
def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref_img', type=str, default='./data/Images/*/00.jpg')
    parser.add_argument('--ref_mask', type=str, default='./data/Annotations/*/00.png')
    parser.add_argument('--img_dir', type=str, default='./data/Images/*')
    parser.add_argument('--out_dir', type=str, default='output')

    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--experiment', type=str, default='single')
    parser.add_argument('--norm', type=bool, default=False)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_arguments()

    experiment_name = args.experiment
    should_normalize = args.norm

    print("Loading SAM...")
    # Load the predictor
    predictor = load_predictor(sam_type=args.sam_type)

    rmrf(args.out_dir)
    mkdirp(args.out_dir)

    for ref_img_path,ref_mask_path,test_img_dir,output_dir in load_dirs(args.ref_img,args.ref_mask,args.img_dir,args.out_dir):
        print(f"Processing {test_img_dir}...")
        persam_f(predictor,ref_img_path,ref_mask_path,test_img_dir,output_dir,experiment_name,should_normalize)

    print("Done!")