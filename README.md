# LoRA-Instruction-Finetune
Instruction finetuning models using LoRA

## How to run with SBATCH
1. Get into a compute node
2. Submit the job with needed configurations. For example: 
```bash
sbatch --job-name=finetune_lora_expanded --gres=gpu:A6000:4 --mem=80G -t 2-00:00:00 --output=.slurm_logs/finetune_lora_expanded.out --mail-type=ALL --mail-user=xinyuel4@andrew.cmu.edu run_finetune_expanded.sh
```