from roboflow import Roboflow

rf = Roboflow()
project = rf.workspace("roboflow-4rfmv").project("climbing-y56wy")
dataset = project.version(6).download("coco-segmentation")

from ..models import ImageEncoderConfig,MaskDecoderConfig
from ..cfg import Config,DataConfig,ModelConfig,TrainConfig

cfg = Config(
    data=DataConfig(
        cls_ids=[1,2,3],
        tasks=["point","box"],
        train_size=None,
        valid_prompts=200,
        points_per_mask=[1,10,10],
    ),
    model=ModelConfig(
        size="vit_t",
        encoder=ImageEncoderConfig(
            use_patch_embed=False
        ),
        decoder=MaskDecoderConfig(
            use_lora=False,
            lora_r=1
        ),
    ),
    train=TrainConfig(
        initial_lr=8e-4,
        cache_embeddings=True,
        run_grad=False
    )
)