docker build --build-arg WANDB_API_KEY=$WANDB_API_KEY --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY --build-arg AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION -f dockerfiles/Dockerfile.smart_poly -t smart_poly .

docker run --gpus all --rm -e DATASET_ID=uoSkoYmmadyjMr7bNwj7 -e ALWAYS_EXPORT=true -it smart_poly