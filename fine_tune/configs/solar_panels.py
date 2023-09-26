
from roboflow import Roboflow
rf = Roboflow(api_key="XDkHpmfoqcJNYiSR1tHD")
project = rf.workspace("optimaize").project("panneau-dataset")
dataset = project.version(47).download("coco-segmentation")

tasks = ["point","box"]

points_per_mask = 3

train_size = 30 # images
create_valid = True

from ..models import ImageEncoderConfig,MaskDecoderConfig
from ..cfg import Config,DataConfig,ModelConfig,TrainConfig

cfg = Config(
    data=DataConfig(
        # cls_ids=[1,2,3],
        tasks=["point","box"],
        # train_size=50,
        # train_prompts=4_000,
        # valid_prompts=200,
        points_per_mask=10,
        create_valid=True,
        check_for_overlap=False
    ),
    model=ModelConfig(
        size="vit_t",
        encoder=ImageEncoderConfig(
            use_patch_embed=False
        ),
        decoder=MaskDecoderConfig(
            use_decoder_lora=True,
            decoder_lora_r=8,
            use_cls=True,
            custom_hypers=True,
            ft=True
        ),
        # binarize_dynamic=True
    ),
    train=TrainConfig(
        initial_lr=2e-4,
        cache_embeddings=True,
        run_grad=True,
        max_steps=12_000,
        loss_scales={
            "focal": 20,
            "mse": 0,
            "dice": 1,
            "ce": 1,
            "cls_loss": 1,
        },
        eval_period=1_500,
        max_epochs=1,
        num_refinement_steps=3,
    )
)