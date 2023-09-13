from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard

from .models import ImageEncoderConfig,MaskDecoderConfig

from typing import List,Optional,Union,Dict,Literal

@dataclass
class DataConfig(YAMLWizard):
    create_valid:bool = False
    cls_ids:Optional[List[int]] = None # any

    train_size:Optional[int] = None # images
    train_prompts:Optional[int] = None

    # number of classes in the dataset - must be set before loading your models.
    num_classes: int = None

    valid_size:Optional[int] = None # images
    valid_prompts:Optional[int] = 1_000

    use_masks:bool = True

    grow_masks:bool = False
    growth_radius:int = 15

    # train on semantic segmentation or point/box-to-mask?
    tasks:List[str] = field(default_factory=lambda: ["point","box"])

    points_per_mask:int = 1
    # for automatic rebalancing, how much can we upsample?
    max_points_per_mask:int = 10

    points_per_side:int = 20
    points_per_img:int = 50

    dataset_name:Optional[str] = None # optional, used only for future reference

DataConfig(cls_ids=None)

@dataclass
class ModelConfig(YAMLWizard):

    encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    decoder: MaskDecoderConfig = field(default_factory=MaskDecoderConfig)

    size:str = "vit_t"

    # try to find a better threshold for binarization (might help with gridiron artifacts)
    binarize_dynamic:str = "false"

    # resolution for onnx mask upscaling
    out_res:int = 512

@dataclass
class TrainConfig(YAMLWizard):
    # use gradient descent or not?
    run_grad:bool = True


    # Optimizer
    initial_lr:float = 2e-4
    weight_decay:float = 0.1

    warmup_steps:int = 500
    max_steps:int = 10_000
    max_epochs:int = 10

    batch_size:int = 5 # currently, I use gradient accumulation for this--they do 256 images per batch.

    log_period:int = 200 # a few batches
    eval_period:int = 500
    wandb_log_period:int = 20

    lr_decay_steps:List[int] = field(default_factory=lambda:[2/3., 0.95])

    lr_decay_factor:float = 0.1


    # loss weightings
    loss_scales:Dict[str,int] = field(default_factory=lambda: {
        "focal": 20,
        "mse": 1,
        "dice": 1,
        "ce": 1,
        "cls_loss": 1,
    })

    benchmark_clicks:List[int] = field(default_factory=lambda: [1,2,3,4])

    # use only losses for the cls tokens (disables loss for single-mask/multimask tokens)
    only_cls_loss:bool = True

    sam_hq_loss:bool = False

    # add multi refinement steps? (i.e. refine the mask with corrective clicks multiple times)
    num_refinement_steps:int = 0

    # cache embeddings for faster training?
    cache_embeddings:bool = True

    # start cls tokens with the same weights as the single-mask token?
    warm_start:bool = True

    use_cuda:bool = True

    # exports
    export_full_decoder:bool = True
    export_full:bool = False

    always_export:bool = True

@dataclass
class WandbConfig(YAMLWizard):
    name:Optional[str] = None
    group:Optional[str] = None

@dataclass
class Config(YAMLWizard):
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)