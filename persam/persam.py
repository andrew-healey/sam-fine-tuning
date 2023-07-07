from common import *
from load import *

def persam(predictor:SamPredictor, ref_img_path:str,ref_mask_path:str,test_img_dir:str,output_dir:str):
    ref_img,ref_mask = load_image(predictor,ref_img_path,ref_mask_path)

    should_normalize = True
    target_feat,target_embedding = get_mask_embed(predictor,ref_mask,should_normalize)

    for test_img_name,test_img_path in load_images_in_dir(test_img_dir):
        load_image(predictor,test_img_path)

        sim_map = get_sim_map(predictor,target_feat)
        attn_sim = sim_map_to_attn(sim_map)
        points = sim_map_to_points(sim_map)

        kwargs = points_to_kwargs(points)
        target_guidance = {
            "attn_sim":attn_sim,  # Target-guided Attention
            "target_embedding":target_embedding  # Target-semantic Prompting
        }

        mask = predict_mask_refined(predictor,target_guidance,**kwargs)

        mask_path = os.path.join(output_dir,test_img_name+".png")
        save_mask(mask,mask_path)
        print("Saved mask to",mask_path)

