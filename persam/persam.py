from common import *
from load import *

def persam(predictor:SamPredictor, ref_img_path:str,ref_mask_path:str,test_img_dir:str,output_dir:str):

    print("Loading reference image...")
    ref_img,ref_mask = load_image(predictor,ref_img_path,ref_mask_path)

    should_normalize = True
    target_feat,target_embedding = get_mask_embed(predictor,ref_mask,should_normalize)

    for test_img_name,test_img_path in load_images_in_dir(test_img_dir):
        print(f"Processing {test_img_name}...")
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

import argparse
def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref_img', type=str, default='./data/Images/backpack_dog/00.jpg')
    parser.add_argument('--ref_mask', type=str, default='./data/Annotations/backpack_dog/00.png')
    parser.add_argument('--img_dir', type=str, default='./data/Images/backpack_dog')
    parser.add_argument('--out_dir', type=str, default='persam')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = get_arguments()

    print("Loading SAM...")
    # Load the predictor
    predictor = load_predictor(sam_type=args.sam_type)

    persam(predictor,args.ref_img,args.ref_mask,args.img_dir,args.out_dir)

    print("Done!")