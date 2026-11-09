#!/bin/bash
# Runs INSIDE the container of train_yolo_pose.slurm, never start it directly on the DGX.
# Ultralytics (the same version as on the laptop) is installed into the mounted project
# folder; pip skips it in later jobs. The container has no graphics libraries, so OpenCV
# is replaced by the headless build.
set -e

# In the container only the mount /workspace is writable, the home directory under its
# real path (/raid/homes/<user>) is read-only. pip --user, its cache and the settings of
# Ultralytics therefore go to a home directory in the mounted project folder. The
# installation stays there, so later jobs start faster.
export HOME=/workspace/parking_training/.container_home
export PYTHONUSERBASE="${HOME}/.local"
export YOLO_CONFIG_DIR="${HOME}/.config/Ultralytics"
export MPLCONFIGDIR="${HOME}/.config/matplotlib"
mkdir -p "${PYTHONUSERBASE}" "${YOLO_CONFIG_DIR}" "${MPLCONFIGDIR}"

MODEL=${MODEL:-yolo26n-pose.pt}
IMGSZ=${IMGSZ:-800}
EPOCHS=${EPOCHS:-150}

echo "Installiere Ultralytics..."
# The PyTorch of the container (2.2) is built against NumPy 1.x, NumPy 2 breaks it
# ("Numpy is not available"). OpenCV < 4.11 still supports NumPy 1.x.
pip install --user --quiet --no-warn-script-location \
    ultralytics==8.4.193 "numpy<2" "opencv-python<4.11"
pip uninstall --quiet -y opencv-python || true
pip install --user --quiet --no-warn-script-location \
    "opencv-python-headless<4.11" "numpy<2"
python3 -c "import numpy, ultralytics, cv2, torch; print('numpy', numpy.__version__, '| ultralytics', ultralytics.__version__, '| opencv', cv2.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available()); assert numpy.__version__.startswith('1.'), 'NumPy 2 installed'"

echo "Starte Training: ${MODEL}, ${IMGSZ} px, ${EPOCHS} Epochen"
cd /workspace/parking_training
python3 train_yolo_pose.py dataset --model "${MODEL}" --imgsz "${IMGSZ}" \
    --epochs "${EPOCHS}" --device 0 --workers 4
