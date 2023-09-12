export DATASET_ID=uoSkoYmmadyjMr7bNwj7

for i in {0..4}
do
python -m fine_tune.aws_smart_poly_config --name binarize --binarize_dynamic=true --group masks
python -m fine_tune.aws_smart_poly_config --name binarize_eval --binarize_dynamic=eval --group masks
python -m fine_tune.aws_smart_poly_config --name no_binarize --binarize_dynamic=false --group masks

python -m fine_tune.aws_smart_poly_config --name lora_4 --use_decoder_lora --group lora
python -m fine_tune.aws_smart_poly_config --name lora_8 --use_decoder_lora --decoder_lora_r=8 --group lora
python -m fine_tune.aws_smart_poly_config --name lora_0 --group lora

python -m fine_tune.aws_smart_poly_config --name no_custom_hypers --group hypers
python -m fine_tune.aws_smart_poly_config --name no_custom_hypers --group hypers --no_custom_hypers

done