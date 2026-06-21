# Virtual Environment Configuration for CLIP-CC-Bench

This directory contains setup scripts, activation scripts, and requirements files for the VLM encoder virtual environments used in CLIP-CC-Bench.

## Overview

Virtual environments are created in the project root directory with the naming convention `venv_{model_name}`. All setup and activation scripts use relative paths for portability across systems.

## Directory Structure

```
clip-cc-bench/
├── venv_configs/                   # This directory
│   ├── README.md                   # This file
│   ├── setup_*.sh                  # Setup scripts
│   ├── activate_*.sh               # Activation scripts
│   └── *-requirements.txt          # Frozen requirements files
├── venv_gte/                       # GTE environment (created by setup scripts)
├── venv_kalm/                      # KaLM environment
├── venv_nemo/                      # Nemo environment
├── venv_nv-embed/                  # NV-Embed environment
└── venv_qwen/                      # Qwen environment
```

## Available Models

| Model | Venv Name | Approximate Size |
|-------|-----------|------------------|
| GTE | venv_gte | ~7 GB |
| KaLM | venv_kalm | ~7 GB |
| Nemo (Nemotron) | venv_nemo | ~7 GB |
| NV-Embed | venv_nv-embed | ~7 GB |
| Qwen | venv_qwen | ~7 GB |

## Setup

### Creating a Single Environment

From the project root:

```bash
# Create GTE environment
bash venv_configs/setup_gte_env.sh

# Create KaLM environment
bash venv_configs/setup_kalm_env.sh

# Create Nemo environment
bash venv_configs/setup_nemo_env.sh

# Create NV-Embed environment
bash venv_configs/setup_nv-embed_env.sh

# Create Qwen environment
bash venv_configs/setup_qwen_env.sh
```

### Creating All Environments

From the project root:

```bash
bash venv_configs/setup_all_environments.sh
```

This will create all 5 environments sequentially (1-3 hours).

## Activation

From the project root:

```bash
# Activate GTE environment
source venv_configs/activate_gte_env.sh

# Activate KaLM environment
source venv_configs/activate_kalm_env.sh

# Activate Nemo environment
source venv_configs/activate_nemo_env.sh

# Activate NV-Embed environment
source venv_configs/activate_nv-embed_env.sh

# Activate Qwen environment
source venv_configs/activate_qwen_env.sh
```

To deactivate:
```bash
deactivate
```

## Requirements Files

Each model has a frozen requirements file (`{model}-requirements.txt`) containing exact package versions for reproducibility. These were generated using `pip freeze` from verified working environments.

## Troubleshooting

### Environment Creation Fails

- Check disk space: `df -h .`
- Check Python version: `python3 --version` (requires 3.8+)
- Verify requirements file exists in `venv_configs/`

### Import Errors After Activation

- Verify activation: `echo $VIRTUAL_ENV`
- Check installed packages: `pip list`
- Reinstall from requirements: `pip install -r venv_configs/{model}-requirements.txt`

### GPU/CUDA Issues

- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Check CUDA version: `nvidia-smi`
- Verify PyTorch CUDA version matches system CUDA

## Cleanup

To remove an environment:

```bash
rm -rf venv_{model_name}
```

To remove all environments:

```bash
rm -rf venv_*
```

## Notes

- All scripts use relative paths for portability
- Virtual environments are created in the project root
- Requirements files contain exact versions for reproducibility
- Each environment is isolated with its own dependencies
