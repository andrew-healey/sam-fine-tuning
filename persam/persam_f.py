from common import *
from load import *

NEG_EXPERIMENT_VALUES = ATTN_EXPERIMENT_VALUES = EMBED_EXPERIMENT_VALUES = BOX_EXPERIMENT_VALUES = NORM_EXPERIMENT_VALUES = [
    True,
    False,
]

FT_EXPERIMENT_NAMES = [
    "sam_embedding",  # cosine similarity of SAM embeddings
    "clip_embedding",  # cosine similarity of CLIP embeddings
    "area",  # closest log mask area
    "bbox_area",  # closest log bbox area
    "perimeter",  # closest log bbox perimeter
    "best_idx",  # Use the best idx from the ref inference
    "best_idx_iou",  # Use the best idx from the ref inference
    "linear_combo",  # Linear combination of SAM embeddings
    "single",  # Recreates PerSAM
    "sim",  # Max avg similarity
    "max_score",  # Mask with highest score
]

from tqdm import tqdm

def persam_f(
    predictor: SamPredictor,
    ref_img_dir: str,
    ref_mask_dir: str,
    test_img_dir: str,
    output_dir: str,
    experiment_name: str,
    should_normalize: bool,
    use_box: bool,
    use_attn: bool,
    use_embed: bool,
    include_neg: bool,
    sim_probe: bool,
):
    if experiment_name not in FT_EXPERIMENT_NAMES:
        raise ValueError(f"Invalid experiment name {experiment_name}")

    print("Loading reference images...")

    ref_img_paths = [ref_img_path for _,ref_img_path in load_images_in_dir(ref_img_dir)]
    ref_mask_paths = [ref_mask_path for _,ref_mask_path in load_images_in_dir(ref_mask_dir)]

    ref_imgs = []
    # ref_masks = []
    ref_feats = []
    target_feats = []
    target_embeddings = []
    feat_dims = None
    gt_masks = []

    for ref_img_path, ref_mask_path in zip(ref_img_paths, ref_mask_paths):
        ref_img, ref_mask = load_image(predictor, ref_img_path, ref_mask_path)
        ref_imgs.append(ref_img)
        ref_feats.append(predictor.features)

        target_feat, target_embedding, feat_dims = get_mask_embed(predictor,ref_mask,should_normalize)
        target_feats.append(target_feat)
        target_embeddings.append(target_embedding)

        mask_cv2 = cv2.imread(ref_mask_path)
        mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2RGB)
        gt_mask = torch.tensor(mask_cv2)[None, :, :, 0] > 0
        gt_masks.append(gt_mask)

    target_feat = torch.mean(target_feats,dim=0)
    target_embedding = torch.mean(target_embeddings,dim=0)

    ref_feat = torch.stack(ref_feats,dim=0)
    gt_mask = torch.stack(gt_masks,dim=0).to(torch.float)

    if sim_probe:

        right_sized_mask = F.interpolate(gt_mask, size=feat_dims, mode="bilinear")[:,0] > 0
        assert right_sized_mask.shape[0] == gt_mask.shape[0]
        assert right_sized_mask.shape[1:] == feat_dims

        sim_probe = get_linear_probe_weights(ref_feat,target_feat,right_sized_mask)
    else:
        sim_probe = None

    def get_prompts(ref_feat):
        with torch.no_grad():
            sim_map = get_sim_map(ref_feat, target_feat,sim_probe)
        attn_sim = sim_map_to_attn(sim_map)
        points = sim_map_to_points(sim_map,include_neg)

        sim_path = os.path.join(output_dir, f"{test_img_name}_sim.png")
        save_mask(sim_map.sigmoid().squeeze(),sim_path)

        kwargs = points_to_kwargs(points)
        target_guidance = {}
        if use_attn:
            target_guidance["attn_sim"] = attn_sim
        if use_embed:
            target_guidance["target_embedding"] = target_embedding
        
        return target_guidance, kwargs, sim_map
    
    target_guidances,kwargss,sim_maps = [],[],[]
    for ref_feat in ref_feats:
        target_guidance, kwargs, sim_map = get_prompts(ref_feat)
        target_guidances.append(target_guidance)
        kwargss.append(kwargs)
        sim_maps.append(sim_map)

    if experiment_name in ["linear_combo", "best_idx", "best_idx_iou"]:
        logit_weights = get_logit_weights(
            predictor, gt_mask, experiment_name, target_guidances, **kwargs
        )
    
    # Not implemented for >1-shot
    elif experiment_name == "clip_embedding":
        raise NotImplementedError()
    elif experiment_name in ["area", "bbox_area", "perimeter"]:

        raise NotImplementedError()

        # Resize gt_mask
        masks, _,_ = predictor.predict(
            multimask_output=True,
        )
        mask_dims = masks.shape[-2:]
        right_sized_mask = F.interpolate(gt_mask, size=mask_dims, mode="bilinear")[:,0] > 0
        assert right_sized_mask.shape[0] == gt_mask.shape[0]
        assert right_sized_mask.shape[1:] == mask_dims

        if experiment_name == "area":

            ref_area = torch.sum(right_sized_mask > 0)
        elif experiment_name in ["bbox_area", "perimeter"]:
            box = mask_to_box(right_sized_mask.cpu().detach().numpy())[0]
            width = box[2] - box[0]
            height = box[3] - box[1]
            ref_area = width * height
            ref_perimeter = 2 * (width + height)

    mkdirp(output_dir)

    raw_img_pairs = load_images_in_dir(test_img_dir)
    for test_img_name, test_img_path in tqdm(raw_img_pairs):
        if(should_log): print(f"Processing {test_img_name}...")
        load_image(predictor, test_img_path)

        target_guidance, kwargs, sim_map = get_prompts()

        # Experiments!
        mask_picking_data = None
        if experiment_name in ["linear_combo", "best_idx", "best_idx_iou"]:
            mask_picking_data = logit_weights
        elif experiment_name == "sam_embedding":
            mask_picking_data = (target_embedding, should_normalize)
        elif experiment_name == "sim":
            mask_picking_data = sim_map
        
        # Not implemented for >1-shot
        elif experiment_name in ["area", "bbox_area"]:
            raise NotImplementedError()
            mask_picking_data = ref_area
        elif experiment_name == "perimeter":
            raise NotImplementedError()
            mask_picking_data = ref_perimeter
        elif experiment_name == "clip_embedding":
            raise NotImplementedError()

        mask,mask_dict = predict_mask_refined(
            predictor,
            target_guidance,
            experiment_name,
            mask_picking_data,
            use_box,
            **kwargs,
        )

        for k,v in mask_dict.items():
            save_mask(v,os.path.join(output_dir,f"{test_img_name}_{k}.png"))

        mask_path = os.path.join(output_dir, test_img_name + ".png")
        save_mask(mask, mask_path)
        if should_log: print("Saved mask to", mask_path)


import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as TVF

lr = 1e-3
train_epoch = 1000
log_epoch = 200

resolution = [256, 256]

def get_logit_weights(
    predictor: SamPredictor,
    gt_mask: torch.Tensor,
    experiment_name: str,
    target_guidance: Dict[str, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    raise NotImplementedError()
    kwargs = {
        **kwargs,
        "high_res": True,
    }

    # Simulated first-step prediction
    masks, scores, logits, original_logits_high = predictor.predict(
        **kwargs,
        # **target_guidance,
        multimask_output=True,
    )

    original_logits_high = TVF.resize(original_logits_high, resolution)
    original_logits_high = original_logits_high.flatten(1)

    gt_mask = TVF.resize(gt_mask.float(), resolution)
    gt_mask = gt_mask.flatten(1).cuda()

    masks = TVF.resize(torch.from_numpy(masks), resolution).flatten(1).cuda()

    # Figure out which logit/mask combination to use.

    # Experiment 1. Find the mask with the highest IoU (or maybe loss) with the target mask.
    # Experiment 2. Run the usual gradient descent on linear weightings of the logits.

    if experiment_name == "linear_combo":
        print("======> Start Training")
        # Learnable mask weights
        mask_weights = Mask_Weights().cuda()
        mask_weights.train()

        optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch)

        for train_idx in range(train_epoch):
            # Weighted sum three-scale masks
            weights = torch.cat(
                (1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights),
                dim=0,
            )
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
                print("Train Epoch: {:} / {:}".format(train_idx, train_epoch))
                current_lr = scheduler.get_last_lr()[0]
                print(
                    "LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}".format(
                        current_lr, dice_loss.item(), focal_loss.item()
                    )
                )

        mask_weights.eval()
        weights = torch.cat(
            (1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0
        ).squeeze(-1)

        return weights

    best_idx = -1

    if experiment_name == "best_idx":
        dice_losses = [
            calculate_dice_loss(original_logits_high[None, idx], gt_mask)
            for idx in range(3)
        ]
        focal_losses = [
            calculate_sigmoid_focal_loss(original_logits_high[None, idx], gt_mask)
            for idx in range(3)
        ]
        losses = [dice_losses[idx] + focal_losses[idx] for idx in range(3)]
        best_idx = torch.argmin(torch.stack(losses))

    elif experiment_name == "best_idx_iou":
        intersections = [
            torch.logical_and(masks[idx], gt_mask).sum() for idx in range(3)
        ]
        unions = [torch.logical_or(masks[idx], gt_mask).sum() for idx in range(3)]
        ious = [intersections[idx] / (unions[idx] + 1e-10) for idx in range(3)]
        best_idx = torch.argmax(torch.stack(ious))

    assert best_idx != -1, "Invalid experiment name"

    return F.one_hot(best_idx, 3).float().cuda()


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def calculate_dice_loss(inputs, targets, num_masks=1):
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


def calculate_sigmoid_focal_loss(
    inputs, targets, num_masks=1, alpha: float = 0.25, gamma: float = 2
):
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

probe_lr = 3e-2
probe_train_epoch = 1000
probe_log_epoch = 200
eps = 1e-10

def get_linear_probe_weights(
        ref_feat: torch.Tensor, # Shape (N, C, H, W)
        target_feat: torch.Tensor, # Shape (C,)
        gt_mask: torch.Tensor, # Shape (N, H, W)
)-> torch.Tensor:
 
    gt_mask = gt_mask.flatten().cuda()

    # convert to (N, HW,C)
    ref_feat = ref_feat.flatten(-2).permute(0,2,1)
    ref_feat = ref_feat / (eps + ref_feat.norm(dim=1,keepdim=True))
    N,HW,C = ref_feat.shape

    assert ref_feat.shape[:2] == gt_mask.shape,f"Shape mismatch: {ref_feat.shape} vs {gt_mask.shape}"
    assert ref_feat.device == gt_mask.device, f"Device mismatch: {ref_feat.device} vs {gt_mask.device}"

    # Learn a (C,) vector of weights which should make attn_sim look like gt_mask
    probe = LinearSimilarityProbe(C,target_feat).cuda()

    optimizer = torch.optim.Adam(probe.parameters(), lr=probe_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=probe_train_epoch)
    probe.train()

    for epoch in range(probe_train_epoch):
        optimizer.zero_grad()
        loss,dice,focal = probe(ref_feat, gt_mask)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % probe_log_epoch == 0:
            print(f"Epoch {epoch} loss: {loss.item()} dice: {dice.item()} focal: {focal.item()}")
    probe.eval()
    return probe

hidden_channels = 3
class LinearSimilarityProbe(nn.Module):
    def __init__(self, num_channels: int,target_feat: torch.Tensor):
        super().__init__()
        # shape (C)
        assert target_feat.shape[0] == num_channels

        # Initialize the probe with the target vector
        self.m1 = nn.Linear(num_channels, hidden_channels)
        with torch.no_grad():
            dummy_m1 = torch.zeros_like(self.m1.weight)
            assert dummy_m1.shape == (hidden_channels, num_channels)
            dummy_m1[0] = target_feat
            self.m1.weight.copy_(dummy_m1)
        self.m2 = nn.Linear(hidden_channels, 1)

        # self.weights = nn.Parameter(torch.ones(num_channels, requires_grad=True) / num_channels)
        # self.bias = nn.Parameter(torch.zeros(1, requires_grad=True))
    def forward(self,
                feat: torch.Tensor, # shape (N,HW, C)
                gt_mask: Optional[torch.Tensor]=None, # shape (N,HW)
                )->torch.Tensor: # shape (,)
        hidden = F.relu(self.m1(feat)) # shape (N,HW, hidden_channels)
        sim_map = self.m2(hidden).squeeze(2) # shape (N,HW)
        if not self.training:
            return sim_map

        dice_loss = calculate_dice_loss(sim_map, gt_mask)
        focal_loss = calculate_sigmoid_focal_loss(sim_map, gt_mask)

        return dice_loss + focal_loss,dice_loss,focal_loss


import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ref_img_dir", type=str, default="./data/Images/*/ref")
    parser.add_argument("--ref_mask_dir", type=str, default="./data/Annotations/*/ref")
    parser.add_argument("--img_dir", type=str, default="./data/Images/*")
    parser.add_argument("--out_dir", type=str, default="output")

    parser.add_argument("--sam_type", type=str, default="vit_h")
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

    parser.add_argument("--sim-probe", action="store_true")
    parser.add_argument("--log",action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=3)

    args = parser.parse_args()
    return args

should_log = False

import pdb
if __name__ == "__main__":
    args = get_arguments()

    experiment_name = args.experiment
    should_normalize = args.norm
    use_box = args.box
    use_attn = args.attn
    use_embed = args.embed
    include_neg = args.neg
    sim_probe = args.sim_probe
    should_log = args.log
    hidden_channels = args.hidden_dim

    print("Loading SAM...")
    # Load the predictor
    predictor = load_predictor(sam_type=args.sam_type)

    rmrf(args.out_dir)
    mkdirp(args.out_dir)

    for ref_img_path, ref_mask_path, test_img_dir, output_dir in load_dirs(
        args.ref_img, args.ref_mask, args.img_dir, args.out_dir
    ):
        print(f"Processing {test_img_dir}...")
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
            sim_probe
        )

    print("Done!")
