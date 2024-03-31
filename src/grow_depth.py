import re
import copy
import torch
import json

def create_deep_layer_mappings(model_state_dict, layers_small, layers_large, expand_type='alternate'):
    """
    A function to copy the state dict of the model
    """

    # hack for doubling the layers
    prefix = 'gpt_neox.layers'

    # Map positions is a dict mapping old parameter names to a list of tuples in the form
    # (new_parameter_name, is_layer_a_copy), where is_layer_a_copy is a boolean indicating
    # whether the parameter is a copy of another layer. If the parameter is a copy, then
    # the weights will be set to the identity function, and the biases will be set to 0.

    map_positions, copy_positions = {}, {}

    # Loop through all the keys in the model state dict
    # to find parameters that are in the transformer layers
    # (ones that start with 'gpt_neox.layers')
    for key in model_state_dict:

        if prefix in key:

            # Find the layer index (between 0 and num_layers - 1)
            layer_idx = re.findall("[-\d]+", key)[0]

            # The origin index looks like 'gpt_neox.layers.0'
            origin_idx = prefix + "." + str(int(layer_idx))

                        
            # If the expansion type is 'alternate'
            if expand_type == 'alternate':

                # If the layer index is greater than or equal to layers_large - layers_small,
                # then we modify the index to be original index + (layers_large - layers_small)
                if int(layer_idx) >= layers_large - layers_small:
                    insert_idx = None
                    origin_key = key.replace(origin_idx, prefix + "." + str(int(layer_idx) + layers_large - layers_small))
    
                else:
                    # This transforms the origin index to the new index
                    # Example: 
                    #   0 -> 1
                    #   1 -> 3
                    #   2 -> 5
                    insert_idx = prefix + "." + str(int(layer_idx) * 2 + 1)

                    # The origin key is formed by replacing the origin index with 2*origin index
                    # Example:
                    #   0 -> 0
                    #   1 -> 2
                    #   2 -> 4
                    origin_key = key.replace(origin_idx, prefix + "." + str(int(layer_idx) * 2))
            
            elif expand_type == 'append':
                
                # If the layer index is greater than or equal to layers_large - layers_small,
                # then we set the insert index to None
                if int(layer_idx) >= layers_large - layers_small:
                    insert_idx = None

                else:
                    insert_idx = prefix + "." + str(layers_small + int( layer_idx ))

                origin_key = key

            # Only if a layer is to be inserted
            if insert_idx is not None:
                insert_key = key.replace( origin_idx, insert_idx )
            else:
                insert_key = None
            
            # If the insert key is not None, then we set the map positions to be a list of tuples
            # of all the new positions that the old parameter will be copied to
            if insert_key is not None:
                map_positions[ key ] = [ (origin_key, False), (insert_key, True) ]

            # If the insert key is None, then we set the map positions to be a list of a single tuple
            # of the new position that the old parameter will be copied to
            else:
                map_positions[ key ] = [ (origin_key, False) ]
            
            
            if insert_key is not None:  
                copy_positions[ insert_key ] = (key, False)
                copy_positions[ origin_key ] = (key, True)

            else:
                copy_positions[ origin_key ] = (key, False)

    return map_positions, copy_positions

def deep_matrix_weight(x, is_identical, is_grad, is_avg_sq):
    # x = (n, m), returns y = (2 * n, 2 * m), used for FF layers
    if is_identical:
        y = torch.zeros_like(x)
        if len(y.shape) > 1:
            y.fill_diagonal_(1)
        # print( f"Setting {x.shape} to diagonal matrix {y}" )
        return y
    else:
        return x.detach().clone()


def deep_bias(x, is_identical, is_grad, is_avg_sq):
    # x = (n, ), returns y = (2 * n, ), used for bias weights
    if is_identical:
        return torch.zeros_like(x)
    else:
        return x.detach().clone()


def deep_param(key, weight, is_identical, is_grad, is_avg_sq):

    # Only in GPT2
    # if "c_attn.weight" in key:
    #     return deep_split_matrix_weight(weight, is_identical=is_identical, is_grad=is_grad, is_avg_sq=is_avg_sq)
    
    if 'weight' in key:
        return deep_matrix_weight(weight, is_identical=is_identical, is_grad=is_grad, is_avg_sq=is_avg_sq)
    elif 'bias' in key:
        return deep_bias(weight, is_identical=is_identical, is_grad=is_grad, is_avg_sq=is_avg_sq)


def deep_state_dict(old_state_dict, map_positions):
    # how to insert layers: direct copy, identical copy
    # operator over the blocks: hacked for GPT-3
    new_state_dict = {}
    for key, weight in old_state_dict.items():
        if map_positions.get( key ):
            for (new_key, new_key_copy_flag) in map_positions.get( key ):
                # print( new_key_copy_flag, is_identical, new_key, key )
                # if new_key_copy_flag:
                #     print( f"Copying {key} to {new_key} and setting to identity" )
                new_state_dict[new_key] = deep_param(key, weight, is_identical=new_key_copy_flag, is_grad=False, is_avg_sq=False)
        else:
            new_state_dict[key] = weight.detach().clone()

    return new_state_dict

def expand_layers(model, layers_small, layers_large, expand_type='alternate'):
    """
    Expand the layers of a model in a function preserving manner from 
    `layers_small` number of layers to `layers_large` number of layers.

    Args:
        model (transformers.AutoModelForCausalLM): The language model to expand
        layers_small (int): The number of layers in the current model
        layers_large (int): The number of layers in the expanded model
        expand_type (str): The type of expansion to perform. Options are 'alternate' and 'append'

    Returns:
        model (transformers.AutoModelForCausalLM): The expanded model
    """

    # Check if the expand_type is valid
    if expand_type not in {'alternate', 'append'}:
        raise ValueError(f"expand_type must be 'alternate' or 'append', not {expand_type}")
    
    # Save old model weights in state dict
    old_state_dict = model.state_dict()

    map_positions, copy_positions = create_deep_layer_mappings(old_state_dict, layers_small, layers_large, expand_type)

    # Expand the state dict
    new_state_dict = deep_state_dict(old_state_dict, map_positions)

    # Use a copy of the model to avoid changing the original model
    model = copy.deepcopy(model)

    # Expand the number of layers from layers_small to layers_large
    layers = model.gpt_neox.layers

    for i in range(layers_large - layers_small):
        if expand_type == 'alternate':
            # Add layers_large - layers_small number of layers alternating between the original layers
            # Example: [0, 1, 2] -> [0, 0, 1, 1, 2] for layers_small = 3 and layers_large = 5
            layers.insert(i * 2, copy.deepcopy(layers[i * 2]))
            
        elif expand_type == 'append':
            # Add layers_large - layers_small number of layers to the end of the model
            # Example: [0, 1, 2] -> [0, 1, 2, 0, 1] for layers_small = 3 and layers_large = 5
            layers.append(copy.deepcopy(layers[i]))
        
        
    # Change the number of layers stored in the config to layers_large
    model.config.num_hidden_layers = layers_large
    
    # Use the saved state dict to preserve the functionality of the original model
    # Load the new_state_dict
    model.load_state_dict(new_state_dict)

    return model


if __name__ == '__main__':
    from transformers import GPTNeoXForCausalLM, AutoTokenizer

    model_70m = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        cache_dir="../.cache/pythia-70m",
    )

    tokenizer_70m = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        cache_dir="../.cache/pythia-70m",
    )

    print(f"Original model: {model_70m.config.num_hidden_layers}")
    inputs = tokenizer_70m("Finish the following sentence:\nRaindrops on roses", return_tensors="pt")
    tokens = model_70m.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))

    model_70m_expanded = expand_layers(model_70m, 6, 8, expand_type='alternate')
    print(f"Expanded model (alternate): {model_70m_expanded.config.num_hidden_layers}")
    inputs = tokenizer_70m("Finish the following sentence:\nRaindrops on roses", return_tensors="pt")
    tokens = model_70m_expanded.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))

    model_70m_expanded = expand_layers(model_70m, 6, 8, expand_type='append')
    print(f"Expanded model (append): {model_70m_expanded.config.num_hidden_layers}")
    inputs = tokenizer_70m("Finish the following sentence:\nRaindrops on roses", return_tensors="pt")
    tokens = model_70m_expanded.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))

    model_70m_expanded = expand_layers(model_70m, 6, 12, expand_type='alternate')
    print(f"Expanded model (alternate): {model_70m_expanded.config.num_hidden_layers}")
    inputs = tokenizer_70m("Finish the following sentence:\nRaindrops on roses", return_tensors="pt")
    tokens = model_70m_expanded.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))

    model_70m_expanded = expand_layers(model_70m, 6, 12, expand_type='append')
    print(f"Expanded model (append): {model_70m_expanded.config.num_hidden_layers}")
    inputs = tokenizer_70m("Finish the following sentence:\nRaindrops on roses", return_tensors="pt")
    tokens = model_70m_expanded.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))

