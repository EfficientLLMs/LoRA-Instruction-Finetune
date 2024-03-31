import torch
from src.lemon_grow_width import *
from transformers.activations import GELUActivation




def test_wide_param_nonlinear():

    class MLP(nn.Module):
        """
        MLP with 1 layer
        """
        
        def __init__(self, hidden_size):
            super(MLP, self).__init__()

            self.mlp_dense_h_to_4h = nn.Linear(hidden_size, hidden_size, bias=True)
            self.act = GELUActivation()
            self.mlp_dense_4h_to_h = nn.Linear(hidden_size, hidden_size, bias=True)

        def forward(self, x):
            x = self.mlp_dense_h_to_4h(x)
            x = self.act(x)
            x = self.mlp_dense_4h_to_h(x)
            return x

    class NeuralNet(nn.Module):
        """
        Neural Net with 1 layer
        """
        
        def __init__(self, in_features, hidden_size, out_features):
            super(NeuralNet, self).__init__()

            self.embed_in = nn.Embedding(in_features, hidden_size)

            # 2 layers of MLP
            self.mlp1 = MLP(hidden_size)
            self.mlp2 = MLP(hidden_size)


            self.final_layer_norm = nn.LayerNorm(hidden_size)
            self.embed_out = nn.Linear(hidden_size, out_features)

        def forward(self, x):
            # print('-'*50)
            # print(f'x: {x}')
            x = self.embed_in(x)
            # print(f'After embed_in: {x}')

            x = self.final_layer_norm(x)
            x = self.mlp1(x)
            print(f'After mlp1: {x}')
            # x = x + self.mlp2(x)
            # print(f'After mlp2: {x}')
            # x = x + self.mlp_dense_4h_to_h(self.act(self.mlp_dense_h_to_4h(x)))
            # print(f'Output before LN: {x}')
            # x = self.final_layer_norm(x)
            x = self.embed_out(x)
            # print(f'embed_out: {x}')
            # print('-'*50)
            return x

    vocab = 3
    old_width = 5
    new_width = 10

    net = NeuralNet(vocab, old_width, vocab)

    # Input for the forward pass
    x = torch.tensor([0, 1, 2])

    # Forward pass on the original network
    y = net(x)

    print(f"Original network with width {old_width}:\n{net}")
    print(f"Output of the original network:\n{y}")

    # All the parameters of the network
    # print(f"dense_4h_to_h weight of the original network:\n{net.state_dict()['mlp_dense_4h_to_h.weight']}")
    # print(f"dense_4h_to_h bias of the original network:\n{net.state_dict()['mlp_dense_4h_to_h.bias']}")

    # Expand the network
    net_wide = NeuralNet(vocab, new_width, vocab)
    new_state_dict = wide_state_dict(net.state_dict(), old_width, new_width)
    net_wide.load_state_dict(new_state_dict)

    # Forward pass on the expanded network
    y_wide = net_wide(x)

    print(f"Expanded network with width {new_width}:\n{net_wide}")
    print(f"Output of the expanded network:\n{y_wide}")

    # All the parameters of the expanded network
    # print(f"Parameters of the expanded network:\n{net_wide.state_dict()}")
    # print(f"dense_4h_to_h weight of the expanded network:\n{net_wide.state_dict()['mlp_dense_4h_to_h.weight']}")
    # print(f"dense_4h_to_h bias of the expanded network:\n{net_wide.state_dict()['mlp_dense_4h_to_h.bias']}")




def test_matrix_row_average_expansion():
    """
    Test the row-average expansion function
    """
    x = torch.rand(3, 4)

    # Define the old and new number of rows
    D_S = 3
    D_T = 5

    print(f"Row-average expansion of a matrix of size ({D_S}, 4) to a new size ({D_T}, 4)")

    # Compute the row-average expansion
    y = matrix_row_average_expansion(x, D_S, D_T)
    
    # Print the original and expanded matrix
    print(f"Original matrix:\n{x}")
    print(f"Expanded matrix:\n{y}")
    print()


def test_matrix_row_zero_expansion():
    """
    Test the row-zero expansion function
    """
    x = torch.rand(3, 4)

    # Define the old and new number of rows
    D_S = 3
    D_T = 5

    print(f"Row-zero expansion of a matrix of size ({D_S}, 4) to a new size ({D_T}, 4)")

    # Compute the row-zero expansion
    y = matrix_row_zero_expansion(x, D_S, D_T)
    
    # Print the original and expanded matrix
    print(f"Original matrix:\n{x}")
    print(f"Expanded matrix:\n{y}")
    print()
    

def test_matrix_row_circular_expansion():
    """
    Test the row-circular expansion function
    """

    x = torch.rand(3, 4)

    # Define the old and new number of rows
    D_S = 3
    D_T = 5

    print(f"Row-circular expansion of a matrix of size ({D_S}, 4) to a new size ({D_T}, 4)")

    # Compute the row-circular expansion
    y = matrix_row_circular_expansion(x, D_S, D_T)

    # Print the original and expanded matrix
    print(f"Original matrix:\n{x}")
    print(f"Expanded matrix:\n{y}")
    print()


def test_matrix_row_random_expansion():
    """
    Test the row-random expansion function
    """

    x = torch.rand(3, 4)

    # Define the old and new number of rows
    D_S = 3
    D_T = 5

    print(f"Row-random expansion of a matrix of size ({D_S}, 4) to a new size ({D_T}, 4)")

    # Compute the row-random expansion
    y = matrix_row_random_expansion(x, D_S, D_T)

    # Print the original and expanded matrix
    print(f"Original matrix:\n{x}")
    print(f"Expanded matrix:\n{y}")
    print()


def test_matrix_column_average_expansion():
    """
    Test the column-average expansion function
    """

    # Define a matrix of size (4, 3)
    x = torch.rand(4, 3)

    # Define the old and new number of columns
    D_S = 3
    D_T = 5

    print(f"Column-average expansion of a matrix of size (4, {D_S}) to a new size (4, {D_T})")

    # Compute the column-average expansion
    y = matrix_column_average_expansion(x, D_S, D_T)

    # Print the original and expanded matrix
    print(f"Original matrix:\n{x}")
    print(f"Expanded matrix:\n{y}")
    print()
    

def test_matrix_column_zero_expansion():
    """
    Test the column-zero expansion function
    """

    # Define a matrix of size (4, 3)
    x = torch.rand(4, 3)

    # Define the old and new number of columns
    D_S = 3
    D_T = 5

    print(f"Column-zero expansion of a matrix of size (4, {D_S}) to a new size (4, {D_T})")

    # Compute the column-zero expansion
    y = matrix_column_zero_expansion(x, D_S, D_T)

    # Print the original and expanded matrix
    print(f"Original matrix:\n{x}")
    print(f"Expanded matrix:\n{y}")
    print()


def test_matrix_column_circular_expansion():
    """
    Test the column-circular expansion function
    """

    # Define a matrix of size (4, 3)
    x = torch.rand(4, 3)

    # Define the old and new number of columns
    D_S = 3
    D_T = 5

    print(f"Column-circular expansion of a matrix of size (4, {D_S}) to a new size (4, {D_T})")

    # Compute the column-circular expansion
    y = matrix_column_circular_expansion(x, D_S, D_T)

    # Print the original and expanded matrix
    print(f"Original matrix:\n{x}")
    print(f"Expanded matrix:\n{y}")


def test_matrix_column_random_expansion():
    """
    Test the column-random expansion function
    """

    # Define a matrix of size (4, 3)
    x = torch.rand(4, 3)

    # Define the old and new number of columns
    D_S = 3
    D_T = 5

    print(f"Column-random expansion of a matrix of size (4, {D_S}) to a new size (4, {D_T})")

    # Compute the column-random expansion
    y = matrix_column_random_expansion(x, D_S, D_T)

    # Print the original and expanded matrix
    print(f"Original matrix:\n{x}")
    print(f"Expanded matrix:\n{y}")
    print()


def test_vector_average_expansion():
    """
    Test the row-average expansion function
    """
    x = torch.rand(4)

    # Define the old and new number of rows
    D_S = 4
    D_T = 6

    print(f"Vector average expansion of a vector of size {D_S} to a new size {D_T}")

    # Compute the average expansion
    y = vector_average_expansion(x, D_S, D_T)
    
    # Print the original and expanded matrix
    print(f"Original vector:\n{x}")
    print(f"Expanded vector:\n{y}")
    print()

def test_vector_zero_expansion():
    """
    Test the row-zero expansion function
    """
    x = torch.rand(4)

    # Define the old and new number of rows
    D_S = 4
    D_T = 6

    print(f"Vector zero expansion of a vector of size {D_S} to a new size {D_T}")

    # Compute the zero expansion
    y = vector_zero_expansion(x, D_S, D_T)
    
    # Print the original and expanded matrix
    print(f"Original vector:\n{x}")
    print(f"Expanded vector:\n{y}")
    print()

def test_vector_circular_expansion():
    """
    Test the row-circular expansion function
    """
    x = torch.rand(4)

    # Define the old and new number of rows
    D_S = 4
    D_T = 6

    print(f"Vector circular expansion of a vector of size {D_S} to a new size {D_T}")

    # Compute the circular expansion
    y = vector_circular_expansion(x, D_S, D_T)
    
    # Print the original and expanded matrix
    print(f"Original vector:\n{x}")
    print(f"Expanded vector:\n{y}")
    print()

def test_vector_random_expansion():
    """
    Test the row-random expansion function
    """
    x = torch.rand(4)

    # Define the old and new number of rows
    D_S = 4
    D_T = 6

    print(f"Vector random expansion of a vector of size {D_S} to a new size {D_T}")

    # Compute the random expansion
    y = vector_random_expansion(x, D_S, D_T)
    
    # Print the original and expanded matrix
    print(f"Original vector:\n{x}")
    print(f"Expanded vector:\n{y}")
    print()


if __name__ == '__main__':
    # Reproducibility
    torch.manual_seed(0)
    
    print(f'Testing matrix expansion')
    test_matrix_row_average_expansion()
    test_matrix_row_zero_expansion()
    test_matrix_row_circular_expansion()
    test_matrix_row_random_expansion()
    test_matrix_column_average_expansion()
    test_matrix_column_zero_expansion()
    test_matrix_column_circular_expansion()
    test_matrix_column_random_expansion()

    print(f'Testing vector expansion')
    test_vector_average_expansion()
    test_vector_zero_expansion()
    test_vector_circular_expansion()
    test_vector_random_expansion()

    print(f'Testing widening on non-linear NNs')
    test_wide_param_nonlinear()

    # print('All tests passed!')