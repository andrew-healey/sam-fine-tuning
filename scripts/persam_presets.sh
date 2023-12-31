
data_classes=($(ls -d ./data/Annotations/*/))
if printf '%s\0' "${data_classes[@]}" | grep -F -x -z -- ./data/Annotations/$1/; then
args=$(cat <<-END
    --ref_img_dir ./data/Images/$1/ref
    --ref_mask_dir ./data/Annotations/$1/ref
    --img_dir ./data/Images/$1
    --out_dir ./output
END
)
fi
if [ "$1" = "perseg" ]
then

# Disable globbing
set -f

args=$(cat <<-END
    --ref_img_dir ./data/Images/*/ref
    --ref_mask_dir ./data/Annotations/*/ref
    --img_dir ./data/Images/*
END
)
fi

script_name=${2:-persam/persam.py}

new_args=$(echo "$script_name" $args "${@:3}")
echo $new_args
python $new_args

# Re-enable globbing
set +f