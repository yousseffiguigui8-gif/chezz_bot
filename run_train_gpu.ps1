$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Using global Python (py launcher) for GPU training..."
py -c "import sys, tensorflow as tf, numpy as np; print('Python:', sys.executable); print('TensorFlow:', tf.__version__); print('NumPy:', np.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
py train.py
