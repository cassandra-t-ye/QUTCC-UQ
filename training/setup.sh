# Replace with your root
CONDA_ROOT=/share/apps/software/anaconda3

# Source the conda.sh hook so that `conda activate` works in bash
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
  echo "ERROR: cannot find ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi
conda activate qutcc2

export TMPDIR=/tmp
export XDG_RUNTIME_DIR=/tmp

python -c "import torch; print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count())"