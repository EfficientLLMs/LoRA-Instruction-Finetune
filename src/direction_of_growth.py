from transformers import AutoTokenizer, pipeline
import eval_utils
from grow import grow_depth, grow_width
import copy
import torch

from interpolate import Interpolate

def grow_model_70_to_410(model_70m):

    # Depth from 6 -> 24
    intermediate_grown = grow_depth.expand_layers(model_70m, 6, 12, expand_type='alternate')
    intermediate_grown = grow_depth.expand_layers(intermediate_grown, 12, 24, expand_type='alternate')

    # Width from 512 -> 1024
    model_70m_ft_grown_410m = grow_width.expand_width(intermediate_grown, 512, 1024)

    return model_70m_ft_grown_410m



def get_model_diff(large_model, grown_small_model):
    
    # Find difference between model parameters of large_pretrained_model and small_pretrained_model, 
    # store the difference in a state dict
    model_diff = {}
    for key, value in large_model.state_dict().items():
        model_diff[key] = value - grown_small_model.state_dict()[key]

    return model_diff


if __name__ == "__main__":
    
    small_pretrained_model_cpt = 'EleutherAI/pythia-70m'
    small_finetuned_model_cpt = 'output/70m/no_trainer/'
    large_pretrained_model_cpt = 'EleutherAI/pythia-410m'

    large_finetuned_model_cpt = 'output/410m/no_trainer/'

    # Load the models
    small_pretrained_model = eval_utils.load_model(small_pretrained_model_cpt)
    small_finetuned_model = eval_utils.load_model(small_pretrained_model_cpt, small_finetuned_model_cpt)
    large_pretrained_model = eval_utils.load_model(large_pretrained_model_cpt)
    large_finetuned_model = eval_utils.load_model(large_pretrained_model_cpt, large_finetuned_model_cpt)

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(small_pretrained_model_cpt)

    # Grow the small_pretrained_model and small_finetuned_model
    grown_small_pretrained_model = grow_model_70_to_410(small_pretrained_model)
    grown_small_finetuned_model = grow_model_70_to_410(small_finetuned_model)

    # Find difference between model parameters of large_pretrained_model and small_pretrained_model, 
    # store the difference in a state dict
    model_diff_state_dict = get_model_diff(large_pretrained_model, grown_small_pretrained_model)

    # Create a shifted_large_pretrained_model by adding the model_diff_state_dict to large_pretrained_model
    shifted_large_pretrained_model = copy.deepcopy(large_pretrained_model)
    
    # Add values of model_diff_state_dict to a values of shifted_large_pretrained_model
    new_state_dict = {}
    for key, value in shifted_large_pretrained_model.state_dict().items():
        new_state_dict[key] = value + model_diff_state_dict[key]

    # Update the state dict of shifted_large_pretrained_model
    shifted_large_pretrained_model.load_state_dict(new_state_dict)

    # Test dataloader
    eval_dataloader = torch.load("./data/eval_dataloader.pt")
    
    # 1. Interpolate large_pretrained_model and shifted_large_pretrained_model
    losses, alphas, rouges = Interpolate.interpolate_models_by_weights_loss_rouge(
        large_pretrained_model,
        shifted_large_pretrained_model,
        5,
        eval_dataloader,
        torch.device("cuda"),
        tokenizer
    )

    # Plot the losses
    eval_utils.plot_losses(alphas, losses, save_path="interpolation_loss_lpt_slpt.png")

    # Plot the rouge scores
    eval_utils.plot_losses(alphas, losses, save_path="interpolation_rouge_lpt_slpt.png")


    # 2. Interpolate shifted_large_pretrained_model and large_finetuned_model
    losses, alphas, rouges = Interpolate.interpolate_models_by_weights_loss_rouge(
        shifted_large_pretrained_model,
        large_finetuned_model,
        5,
        eval_dataloader,
        torch.device("cuda"),
        tokenizer
    )

    # Plot the losses
    eval_utils.plot_losses(alphas, losses, save_path="interpolation_loss_slpt_lft.png")

    # Plot the rouge scores
    eval_utils.plot_losses(alphas, losses, save_path="interpolation_rouge_slpt_lft.png")