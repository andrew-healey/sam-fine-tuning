# Disable globbing
set -f

if [ "$1" = "backpack_dog" ]
then
args=$(cat <<-END
    --ref_img ./data/Images/backpack_dog/00.jpg
    --ref_mask ./data/Annotations/backpack_dog/00.png
    --img_dir ./data/Images/backpack_dog
    --out_dir ./output
END
)
fi
if [ "$1" = "perseg" ]
then
args=$(cat <<-END
    --ref_img ./data/Images/*/00.jpg
    --ref_mask ./data/Annotations/*/00.png
    --img_dir ./data/Images/*
END
)
fi

script_name=${2:-persam/persam.py}

new_args=$(echo "$script_name" $args "${@:3}")
echo $new_args
python $new_args
set +f