from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard

from .models import ImageEncoderConfig,MaskDecoderConfig

from typing import List,Optional,Union,Dict

@dataclass
class DataConfig(YAMLWizard):
    create_valid:bool = False
    cls_ids:Optional[List[int]] = None # any

    train_size:Optional[int] = None # images
    train_prompts:Optional[int] = None

    # number of classes in the dataset - must be set before loading your models.
    num_classes: int = None

    valid_size:Optional[int] = None # images
    valid_prompts:Optional[int] = None

    use_masks:bool = True

    grow_masks:bool = False
    growth_radius:int = 15

    # train on semantic segmentation or point/box-to-mask?
    tasks:List[str] = field(default_factory=lambda: ["point","box"])

    points_per_mask:int = 1
    points_per_side:int = 20
    points_per_img:int = 50

    dataset_name:Optional[str] = None # optional, used only for future reference

DataConfig(cls_ids=None)

@dataclass
class ModelConfig(YAMLWizard):

    encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    decoder: MaskDecoderConfig = field(default_factory=MaskDecoderConfig)

    size:str = "vit_h"

@dataclass
class TrainConfig(YAMLWizard):
    # use gradient descent or not?
    run_grad:bool = True


    # Optimizer
    initial_lr:bool = 2e-4
    weight_decay:bool = 0.1

    warmup_steps:int = 500
    max_steps:int = 50_000
    max_epochs:int = 100

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

    benchmark_clicks:List[int] = field(default_factory=lambda: [1,2,5])

    # use only losses for the cls tokens (disables loss for single-mask/multimask tokens)
    only_cls_loss:bool = True

    # add multi refinement steps? (i.e. refine the mask multiple times)
    num_refinement_steps:int = 0

    # cache embeddings for faster training?
    cache_embeddings:bool = True

    # start cls tokens with the same weights as the single-mask token?
    warm_start:bool = True

    use_cuda:bool = True

    # exports
    export_full_decoder:bool = True
    export_full:bool = False

@dataclass
class Config(YAMLWizard):
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)