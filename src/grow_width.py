import torch
import copy

def wide_matrix_dense(x, old_width, new_width):
    """
    Function preserving expansion of linear layer weight matrix
    """

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    new_rows = (x.shape[0] * new_width) // old_width
    new_cols = (x.shape[1] * new_width) // old_width
    y = torch.zeros(new_rows, new_cols)

    print(f'Expanding from {x.shape} to {y.shape}')

    # Copy old matrix into new matrix
    y[:x.shape[0], :x.shape[1]] = x

    # Copy first new_cols-x.shape[1] columns of x into the last new_cols-x.shape[1] columns of y
    y[:x.shape[0], x.shape[1]:] = x[:, :new_cols - x.shape[1]]

    # Copy first new_rows-x.shape[0] rows of x into the last new_rows-x.shape[0] rows of y
    y[x.shape[0]:, :x.shape[1]] = x[:new_rows - x.shape[0], :] / 2
    y[:new_rows - x.shape[0], :x.shape[1]] /= 2

    return y


def wide_bias_dense(x, old_width, new_width):
    """
    Function preserving expansion of linear layer bias vector
    """

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    new_cols = (x.shape[0] * new_width) // old_width
    y = torch.zeros(new_cols)

    # Apply Net2Net expansion
    y[:x.shape[0]] = x
    y[x.shape[0]:] = x[:new_cols - x.shape[0]] / 2
    y[:new_cols - x.shape[0]] /= 2

    return y





# def wide_param(key, weight, old_width, new_width):
#     if 'embed_in' in key:
#         return wide_embedding_in(weight, old_width, new_width)
    

#     elif 'embed_out' in key:
#         if 'bias' in key:
#             return weight
#         elif 'weight' in key:
#             return wide_embedding_out(weight, old_width, new_width)
#     elif 'weight' in key:
#         return wide_matrix_dense(weight, old_width, new_width)
#     elif 'bias' in key:
#         return wide_bias_dense(weight, old_width, new_width)


def wide_param(key, weight, old_width, new_width):

    if 'embed_in' in key:
        return wide_embedding_in(weight, old_width, new_width)
    
    elif 'embed_out' in key:
        if "weight" in key:
            return wide_embedding_out(weight, old_width, new_width)
        elif "bias" in key:
            return weight
    
    elif "layernorm" in key or "layer_norm" in key:
        return wide_bias_dense(weight, old_width, new_width)
    
    elif ("attention" in key or "mlp" in key) and ("query_key_value" in key or "dense" in key):
        if "weight" in key:
            return wide_matrix_dense(weight, old_width, new_width)
        elif "bias" in key:
            return wide_bias_dense(weight, old_width, new_width)
        
    return weight


#     # Not found
#     print(key)
#     import ipdb; ipdb.set_trace()
#     # raise ValueError(key, weight.shape)


def wide_state_dict(old_state_dict, old_width, new_width):
    new_state_dict = {}
    for key, weight in old_state_dict.items():
        # print(f'key: {key}')
        new_state_dict[key] = wide_param(key, weight, old_width, new_width)
    return new_state_dict


def wide_embedding_in(x, old_width, new_width):
    """
    Function preserving expansion of input embedding layer from (vocab_size, old_width) 
    to (vocab_size, new_width)

    Args:
        x (torch.Tensor): input tensor of shape (vocab_size, old_width)
        old_width (int): old width of the embedding layer
        new_width (int): new width of the embedding layer

    Returns:
        torch.Tensor: expanded tensor of shape (vocab_size, new_width)
    """

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    print(f'Expanding from {x.shape} to ({x.shape[0]}, {new_width})')
    y = torch.zeros(x.shape[0], new_width)

    # Apply Net2Net expansion
    y[:, :old_width] = x
    y[:, old_width:] = x[:, :new_width - old_width] / 2
    y[:, :new_width - old_width] /= 2

    return y


def wide_embedding_out(x, old_width, new_width):
    """
    Function preserving expansion of output embedding layer from (old_width, vocab_size) 
    to (new_width, vocab_size)

    Args:
        x (torch.Tensor): input tensor of shape (old_width, vocab_size)
        old_width (int): old width of the embedding layer
        new_width (int): new width of the embedding layer

    Returns:
        torch.Tensor: expanded tensor of shape (new_width, vocab_size)
    """

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    new_rows = x.shape[0]
    new_cols = (x.shape[1] * new_width) // old_width
    y = torch.zeros(new_rows, new_cols)

    print(f'Expanding from {x.shape} to {y.shape}')

    # Copy old matrix into new matrix
    y[:, :x.shape[1]] = x

    # Copy first new_cols-x.shape[1] columns of x into the last new_cols-x.shape[1] columns of y
    y[:, x.shape[1]:] = x[:, :new_cols - x.shape[1]]

    # Copy first new_rows-x.shape[0] rows of x into the last new_rows-x.shape[0] rows of y
    # y[x.shape[0]:, :x.shape[1]] = x[:new_rows - x.shape[0], :] / 2
    # y[:new_rows - x.shape[0], :x.shape[1]] /= 2

    return y

def expand_width(model, old_width, new_width):
    """
    Expand the width of a model in a function preserving model from size `old_width` to 
    `new_width`. 

    Args:
        model (transformers.AutoModelForCausalLM): The language model to expand
        old_width (int): The old width of the model
        new_width (int): The new width of the model

    Returns:
        model (transformers.AutoModelForCausalLM): The expanded model
    """

    # Save old model weights in state dict
    old_state_dict = model.state_dict()

    # Use a copy of the model to avoid changing the original model
    old_config = model.config
    new_config_dict = old_config.to_dict()
    new_config_dict["hidden_size"] = new_width
    new_config_dict["intermediate_size"] = new_width * 4
    new_config = type(old_config).from_dict(new_config_dict)
    
    model = type(model)(new_config)

    # Create new state dict
    new_state_dict = wide_state_dict(old_state_dict, old_width, new_width)

    # Set config hidden size
    # model.config.hidden_size = new_width
    # model.config.intermediate_size = new_width * 4

    # Load new state dict
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

    print(f"Original model: {model_70m.config.hidden_size}")
    inputs = tokenizer_70m("Finish the following sentence:\nRaindrops on roses", return_tensors="pt")
    
    print(f'hidden state: {model_70m(**inputs)[0].shape}')
    
    tokens = model_70m.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))

    

    model_70m_wide = expand_width(model_70m, 512, 1024)
    print(f"Expanded model: {model_70m_wide.config.hidden_size}")
    inputs = tokenizer_70m("Finish the following sentence:\nRaindrops on roses", return_tensors="pt")

    print(f'hidden state: {model_70m_wide(**inputs)[0].shape}')

    tokens = model_70m_wide.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))
