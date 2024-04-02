import torch
import torch.nn as nn
import copy


class NeuralNet(nn.Module):
    """
    Neural Net with 1 layer
    """
    
    def __init__(self, in_features, hidden_size, out_features):
        super(NeuralNet, self).__init__()

        self.embed_in = nn.Embedding(in_features, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.embed_out = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = self.embed_in(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.layer_norm(x)
        x = self.embed_out(x)
        return x


def wide_bias(x, old_width, new_width, average=False):
    """
    Function preserving expansion of bias vector from (old_width) 
    to (new_width)

    Args:
        x (torch.Tensor): input tensor of shape (old_width)
        old_width (int): old width of the bias vector
        new_width (int): new width of the bias vector

    Returns:
        torch.Tensor: expanded tensor of shape (new_width)
    """

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    new_cols = (x.shape[0] * new_width) // old_width
    y = torch.zeros(new_cols)

    y[:x.shape[0]] = x
    y[-x.shape[0]:] = x

    # if average:
    #     # Pad with average of all [:x.ahape[0]] neurons
    #     y[:x.shape[0]] = x * (2 - new_cols/x.shape[0])
    #     mean_value = x.mean(dim=0)
    #     y[x.shape[0]:] = mean_value

    #     # Sum of original matrix
    #     print(f'Original sum: {x.sum()}')
    #     print(f'New sum: {y.sum()}')
    #     return y


    # # Apply Net2Net expansion
    # y[:x.shape[0]] = x
    # y[x.shape[0]:] = x[:new_cols - x.shape[0]] / 2
    # y[:new_cols - x.shape[0]] /= 2

    return y

def wide_matrix(x, old_width, new_width, average=False):
    """
    Function preserving expansion of weight matrix from (old_width, old_width) 
    to (new_width, new_width)

    A weight matrix is of shape (out_size, in_size)

    Args:
        x (torch.Tensor): input tensor of shape (old_width)
        old_width (int): old width of the weight matrix
        new_width (int): new width of the weight matrix

    Returns:
        torch.Tensor: expanded tensor of shape (new_width, new_width)
    """

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    new_rows = (x.shape[0] * new_width) // old_width
    new_cols = (x.shape[1] * new_width) // old_width
    y = torch.zeros(new_rows, new_cols)
    
    # Copy old matrix into new matrix
    y[:x.shape[0], :x.shape[1]] = x

    y[-x.shape[0]:, -x.shape[1]:] = x

    # Copy first new_cols-x.shape[1] columns of x into the last new_cols-x.shape[1] columns of y
    # y[:x.shape[0], x.shape[1]:] = x[:, :new_cols - x.shape[1]]

    # Average 
    # if average:
    #     # Pad with average of all [:x.ahape[0]] neurons
    #     y[:x.shape[0], x.shape[1]:] *= (2 - new_cols/x.shape[0])

    #     mean_value = y[x.shape[0]:, :x.shape[1]].mean(dim=0)
    #     y[x.shape[0]:, :x.shape[1]] = mean_value
    #     # y[:new_rows - x.shape[0], :x.shape[1]] *= (2 - new_width/old_width)
    #     return y


    # Copy first new_rows-x.shape[0] rows of x into the last new_rows-x.shape[0] rows of y
    # y[x.shape[0]:, :x.shape[1]] = x[:new_rows - x.shape[0], :] / 2
    # y[:new_rows - x.shape[0], :x.shape[1]] /= 2

    return y

def wide_embedding_in(x, old_width, new_width):
    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    y = torch.zeros(x.shape[0], new_width)

    # Apply Net2Net expansion
    y[:, :old_width] = x
    y[:, old_width:] = x[:, :new_width - old_width] / 2
    y[:, :new_width - old_width] /= 2

    return y

def wide_layer_norm(x, old_width, new_width):

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    new_cols = (x.shape[0] * new_width) // old_width
    y = torch.zeros(new_cols)
 
    eta = torch.floor(new_width/old_width) * old_width/new_width

    # Apply Net2Net expansion
    y[:x.shape[0]] = x
    y[x.shape[0]:] = x[:new_cols - x.shape[0]] / 2
    y[:new_cols - x.shape[0]] /= 2

    return y * eta

def wide_layer_norm_bias(x, old_width, new_width):

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    new_cols = (x.shape[0] * new_width) // old_width
    y = torch.zeros(new_cols)
 
    # Apply Net2Net expansion
    y[:x.shape[0]] = x
    y[x.shape[0]:] = 0

    return y

def wide_embedding_out(x, old_width, new_width):
    
    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    new_rows = x.shape[0]
    new_cols = (x.shape[1] * new_width) // old_width
    y = torch.zeros(new_rows, new_cols)

    # print(f'Expanding from {x.shape} to {y.shape}')

    # Copy old matrix into new matrix
    y[:, :x.shape[1]] = x

    # Copy first new_cols-x.shape[1] columns of x into the last new_cols-x.shape[1] columns of y
    y[:, x.shape[1]:] = x[:, :new_cols - x.shape[1]]

    # Copy first new_rows-x.shape[0] rows of x into the last new_rows-x.shape[0] rows of y
    # y[x.shape[0]:, :x.shape[1]] = x[:new_rows - x.shape[0], :] / 2
    # y[:new_rows - x.shape[0], :x.shape[1]] /= 2

    return y


def wide_state_dict(old_state_dict, old_width, new_width):
    new_state_dict = {}
    for key, weight in old_state_dict.items():
        # print(f'key: {key}')
        new_state_dict[key] = wide_param(key, weight, old_width, new_width)
    return new_state_dict

def wide_param(key, weight, old_width, new_width):
    if 'embed_in' in key:
        return wide_embedding_in(weight, old_width, new_width)
    elif 'embed_out' in key:
        if 'bias' in key:
            return weight
        elif 'weight' in key:
            return wide_embedding_out(weight, old_width, new_width)
        
    elif 'final_layer_norm' in key:
        return wide_bias(weight, old_width, new_width) / 2

    
    elif 'layer_norm' in key or 'layernorm' in key:
        # TODO: fix
        return wide_bias(weight, old_width, new_width)
        # if 'bias' in key:
        #     return wide_layer_norm_bias(weight, old_width, new_width) 
        # elif 'weight' in key:
        #     return wide_layer_norm(weight, old_width, new_width) * old_width / new_width
    
    elif 'weight' in key:
        return wide_matrix(weight, old_width, new_width)
    elif 'bias' in key:
        return wide_bias(weight, old_width, new_width)



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

    # Calculate new number of attention heads as new_width / (old_width/old_n_heads)
    new_n_heads = int(new_width / (old_width / old_config.num_attention_heads))

    new_config_dict["hidden_size"] = new_width
    new_config_dict["intermediate_size"] = new_width * 4
    new_config_dict["num_attention_heads"] = new_n_heads
    new_config_dict["_name_or_path"] += f"-expand-width-{new_width}"
    new_config = type(old_config).from_dict(new_config_dict)
    

    model = type(model)(new_config)

    # # Create new state dict
    new_state_dict = wide_state_dict(old_state_dict, old_width, new_width)

    # Set config hidden size

    # Load new state dict
    model.load_state_dict(new_state_dict)

    return model
    


if __name__ == '__main__':

    # from render_md import render_docstring_html, save_html_file

    # Render the docstring as HTML
    # html_docstring = render_docstring_html(vector_average_expansion)

    # Save the HTML content to a file
    # save_html_file(html_docstring, "docstring.html")

    # Set torch random seed
    torch.manual_seed(1)

    # net = NeuralNet(3, 5, 3)

    # example_input = torch.tensor([[0, 2, 1]]).long()
    
    # # Forward pass
    # out = net(example_input)

    # # Print the output
    # print(f'out: {out}')

    # # State dict
    # old_state_dict = net.state_dict()

    # # for key, weight in old_state_dict.items():
    # #     print(f'key: {key}, weight: {weight.shape}')


    # new_state_dict = expand(old_state_dict, 5, 10)
    
    # net_copy = NeuralNet(3, 10, 3)

    # net_copy.load_state_dict(new_state_dict)

    # # Forward pass
    # out_copy = net_copy(example_input)
    # print(f'out: {out_copy}')

    # print(f'ratio: {out_copy/out}')



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


    inputs = tokenizer_70m("The best place on earth is", return_tensors="pt")

    print(f'hidden state: {model_70m(**inputs)[0].shape}')

    tokens = model_70m.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))



    model_70m_wide = expand_width(model_70m, 512, 1024)
    print(f"Expanded model: {model_70m_wide.config.hidden_size}")
    inputs = tokenizer_70m("Finish the following sentence:\nRaindrops on roses", return_tensors="pt")

    print(f'hidden state: {model_70m_wide(**inputs)[0].shape}')

    tokens = model_70m_wide.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))

    inputs = tokenizer_70m("The best place on earth is", return_tensors="pt")

    print(f'hidden state: {model_70m_wide(**inputs)[0].shape}')

    tokens = model_70m_wide.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))
