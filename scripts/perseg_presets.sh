if $2 == 'backpack_dog' then
args=$(cat <<-END
    --ref_img ./data/Images/backpack_dog/00.jpg
    --ref_mask ./data/Annotations/backpack_dog/00.png
    --img_dir ./data/Images/backpack_dog
END
)
fi
if $2 == 'perseg' then
args=$(cat <<-END
    --ref_img ./data/Images/*/00.jpg
    --ref_mask ./data/Annotations/*/00.png
    --img_dir ./data/Images/*
END
)
fi

python $1 $args ${@:3}