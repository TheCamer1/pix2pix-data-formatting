#!/bin/bash
#SBATCH --job-name=MRtoCT
#SBATCH --account=PRJ-BWsCT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=MRtoCT_%j.out
#SBATCH --mail-type=ALL # Optional: for email notifications

# Load Python module
module load python/3.9.15

# Define the path for the virtual environment
VENV_PATH="$HOME/.virtualenvs/MRCT"

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    python -m venv "$VENV_PATH"
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade pip and install required packages
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url http://download.pytorch.org/whl/cu118 --trusted-host download.pytorch.org
pip install dominate
pip install visdom
pip install wandb

# Ensure the script uses the virtual environment's Python interpreter
python -m train.py --dataroot /rds/PRJ-BWsCT/FormattedFinal --name MRtoCT_pix2pix --model pix2pix --direction AtoB
