conda activate pytorch
export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`

python ../examples/training/nli/training_nli.py
