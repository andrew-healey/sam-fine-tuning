from .cfg import Config
from .models import WrappedMaskDecoder,WrappedSamModel

import torch
import json

import os
def export(
        dir:str,
        cfg:Config,
        sam: WrappedSamModel,
        device:torch.device
        ):

        # write cfg to dir/cfg.yaml
        # uses dataclass_wizard YAMLWizard built-in method
        cfg.to_yaml_file(os.path.join(dir,"cfg.yaml"))

        if cfg.train.export_full_decoder:
            # export full decoder
            torch.save(sam.decoder.state_dict(),os.path.join(dir,"decoder.pt"))
        
        if cfg.train.export_full:
            torch.save(sam.state_dict(),os.path.join(dir,"sam.pt"))
        
        # export trainable state dict
        torch.save(sam.get_trainable_state_dict(),os.path.join(dir,"trainable.pt"))

        # save class list to dir/classes.txt
        with open(os.path.join(dir,"classes.json"),"w") as f:
            classes = cfg.data.classes
            json.dump(classes,f)

        # export onnx
        onnx_export(dir,cfg,sam,device)

import torch

from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

import warnings

cpu = torch.device("cpu")
def onnx_export(dir:str,cfg:Config,sam:WrappedSamModel,device:torch.device):

    og_sam = sam.predictor.model
    og_sam.to(cpu)

    onnx_model_path = f"{dir}/with_classes.onnx"

    onnx_model = SamOnnxModel(
         og_sam,
         return_single_mask=True,
         out_res=cfg.model.out_res,
         use_cls_token=True,
         use_normal_token=False,
    )

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = og_sam.prompt_encoder.embed_dim
    embed_size = og_sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float)
    }
    # output_names = ["masks", "iou_predictions", "low_res_masks"]
    output_names = ["cls_masks","cls_iou_predictions","cls_low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )


    onnx_model_quantized_path = f"{dir}/with_classes_quantized.onnx"
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=onnx_model_quantized_path,
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )

    og_sam.to(device)
    og_sam.prompt_encoder.to(cpu)