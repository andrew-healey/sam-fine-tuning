export DATASET_ID=uoSkoYmmadyjMr7bNwj7
export MAX_IDLE_TIME_SECONDS=3600 # 1 hr of training

# python -m fine_tune.aws_smart_poly_config --name binarize --binarize_dynamic=true --group masks
# python -m fine_tune.aws_smart_poly_config --name binarize_eval --binarize_dynamic=eval --group masks
# python -m fine_tune.aws_smart_poly_config --name no_binarize --binarize_dynamic=false --group masks

# python -m fine_tune.aws_smart_poly_config --name lora_4 --use_decoder_lora --group lora
# python -m fine_tune.aws_smart_poly_config --name lora_8 --use_decoder_lora --decoder_lora_r=8 --group lora
# python -m fine_tune.aws_smart_poly_config --name lora_0 --group lora

python -m fine_tune.aws_smart_poly_config --name custom_hypers --group hypers
python -m fine_tune.aws_smart_poly_config --name no_custom_hypers --group hypers --no_custom_hypers

# python -m fine_tune.aws_smart_poly_config --name inf_prompts --group num_train_prompts
# python -m fine_tune.aws_smart_poly_config --name 5k_prompts --train_prompts=5000 --group num_train_prompts
# python -m fine_tune.aws_smart_poly_config --name 4k_prompts --train_prompts=4000 --group num_train_prompts
# python -m fine_tune.aws_smart_poly_config --name 3k_prompts --train_prompts=3000 --group num_train_prompts
# python -m fine_tune.aws_smart_poly_config --name 2k_prompts --train_prompts=2000 --group num_train_prompts
# python -m fine_tune.aws_smart_poly_config --name 1k_prompts --train_prompts=1000 --group num_train_prompts

# python -m fine_tune.aws_smart_poly_config --name 2_lr --initial_lr=0.0002 --group lrs
# python -m fine_tune.aws_smart_poly_config --name 4_lr --initial_lr=0.0004 --group lrs
# python -m fine_tune.aws_smart_poly_config --name 8_lr --initial_lr=0.0008 --group lrs

# python -m fine_tune.aws_smart_poly_config --name embed --use_patch_embed --group patch_embed
# python -m fine_tune.aws_smart_poly_config --name no_embed --group patch_embed

python -m fine_tune.aws_smart_poly_config --name sam_hq_loss --sam_hq_loss --group sam_hq_loss
python -m fine_tune.aws_smart_poly_config --name normal_loss --group sam_hq_loss

python -m fine_tune.aws_smart_poly_config --name 0_refine --num_refinement_steps=0 --group clicks
python -m fine_tune.aws_smart_poly_config --name 1_refine --num_refinement_steps=1 --group clicks
python -m fine_tune.aws_smart_poly_config --name 2_refine --num_refinement_steps=2 --group clicks
python -m fine_tune.aws_smart_poly_config --name 3_refine --num_refinement_steps=3 --group clicks

python -m fine_tune.aws_smart_poly_config --name 10_pts --group max_pts_per_mask
python -m fine_tune.aws_smart_poly_config --name 20_pts --max_points_per_mask=20 --group max_pts_per_mask

python -m fine_tune.aws_smart_poly_config --name 1_batch --batch_size=1 --group batch_size
python -m fine_tune.aws_smart_poly_config --name 5_batch --batch_size=5 --group batch_size
python -m fine_tune.aws_smart_poly_config --name 20_batch --batch_size=20 --group batch_size
python -m fine_tune.aws_smart_poly_config --name 50_batch --batch_size=50 --group batch_size
python -m fine_tune.aws_smart_poly_config --name 100_batch --batch_size=100 --group batch_size

python -m fine_tune.aws_smart_poly_config --name warm --group warm_start
python -m fine_tune.aws_smart_poly_config --name cold --no_warm_start --group warm_start