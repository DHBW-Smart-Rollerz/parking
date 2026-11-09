#!/bin/bash
#SBATCH --job-name=USERNAME_parkingyolo         # Name des Jobs (#USERNAME_#JOBNAME)
#SBATCH --partition=debug                       # Partition für GPU-Aufgaben
#SBATCH --gres=gpu:user:1                       # Anzahl der GPUs
#SBATCH --cpus-per-task=4                       # Anzahl der CPU-Kerne
#SBATCH --mem=16G                               # Arbeitsspeicher
#SBATCH --time=02:00:00                         # Maximale Laufzeit
#SBATCH --output=slurm-%j.out                   # Ausgabeprotokoll (%j = Job-ID)

# SLURM job (Pyxis container, example 3 of the DGX H100 How To): YOLO pose training of
# the parking spot detector. Expected in ~/parking_training on the DGX:
#   train_yolo_pose.slurm, run_training.sh, train_yolo_pose.py (tools/dgx/)
#   dataset/  (export of notebooks/train_yolo_pose.ipynb, data/yolo/<dataset>/)
# Start:        cd ~/parking_training && sbatch train_yolo_pose.slurm
# Other model:  MODEL=yolo11n-pose.pt sbatch train_yolo_pose.slurm

# Umgebungsvariablen für Container-GPU-Support
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# GPU-Check INNERHALB des Containers
echo "Überprüfe GPU-Sichtbarkeit im Container..."
srun --container-image=pytorch/pytorch:latest nvidia-smi -L

# Eigentlicher Programmaufruf ($HOME is mounted as /workspace)
srun --container-image=pytorch/pytorch:latest \
    --container-mounts="$HOME:/workspace" \
    bash /workspace/parking_training/run_training.sh
