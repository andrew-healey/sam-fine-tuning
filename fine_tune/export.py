from .cfg import Config
from .models import WrappedMaskDecoder,WrappedSamModel

import torch

import os
def export(
        dir:str,
        cfg:Config,
        sam: WrappedSamModel
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
