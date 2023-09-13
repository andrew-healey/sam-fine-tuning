# Download

```bash
git clone https://github.com/andrew-healey/sam-fine-tune.git --recursive
```

# Run locally

```bash
conda env create -n sam python=3.11.5 -c conda-forge
conda activate sam
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
python -m fine_tune.aws_smart_poly_config
```

# Dockerfile

```bash
export AWS_ACCESS_KEY_ID=<your_access_key_id>
export AWS_SECRET_ACCESS_KEY=<your_secret_access_key>
export AWS_DEFAULT_REGION=<your_region>
./roboflow-train/one_publish_train_version.sh
```