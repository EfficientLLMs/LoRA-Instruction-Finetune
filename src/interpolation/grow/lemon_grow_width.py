import torch
import torch.nn as nn






#####################
# LEMON expansions
#####################

#####################
# vector expansions #
#####################

def vector_average_expansion(x, D_S, D_T):
    """
    Computes the vector average expansion of a given input vector `x` of dimension `D_S`
    to a new dimension `D_T`.

    The vector average expansion is defined as:

    x* = Vavg(x) = Concat[x^T, ..., x^T, Avg(x), ..., Avg(x)] ∈ ℝ^Dτ

    where:
    - x ∈ ℝ^Ds is the input vector of dimension Ds
    - Avg(x) = E[x] = (1/Ds) Σ_i^Ds x[i] is the average of x
    - x* is the average expanded vector of dimension Dτ
    - Dτ ≥ Ds, i.e., the new width must be greater than or equal to the old width
    - [Dτ/Ds] is the number of repeated copies of x^T
    - Dτ mod Ds is the number of repeated copies of Avg(x)

    Args:
        x (torch.Tensor(D_S)): The input vector of dimension `D_S`.
        D_S (int): The dimension of the input vector `x`.
        D_T (int): The desired dimension of the output vector.

    Returns:
        torch.Tensor(D_T): The average expanded vector of dimension `D_T`.
    """

    assert D_T >= D_S, "New width must be greater than or equal to D_S"
    assert D_T - D_S <= D_S, "New width must be at most twice the D_S"

    # Compute the average of the input vector
    avg = x.mean()

    # Compute the number of repeated averages needed
    num_avg_repeats = D_T % D_S

    # Compute the number of times to repeat x
    num_repeats = D_T // D_S

    # Concatenate the repeated vector and repeated averages
    y = torch.cat([x.repeat(num_repeats), avg.repeat(num_avg_repeats)])

    return y


def vector_zero_expansion(x, D_S, D_T):
    """
    Computes the vector zero expansion of a given input vector `x` of dimension `D_S`
    to a new dimension `D_T`.

    The vector zero expansion is defined as:

    xzero = Vzero(x) = Concat[x^T, ..., x^T, 0, ..., 0] ∈ ℝ^Dr

    where:
    - x ∈ kamers RDs is the input vector of dimension Ds
    - xzero is the zero expanded vector of dimension Dr with Dr ≥ Ds
    - [Dr/Ds] is the number of repeated copies of x^T
    - Dr mod Ds is the number of zeros appended

    Args:
        x (torch.Tensor(D_S)): The input vector of dimension `D_S`.
        D_S (int): The dimension of the input vector `x`.
        D_T (int): The desired dimension of the output vector.

    Returns:
        torch.Tensor(D_T): The zero expanded vector of dimension `D_T`.
    """

    assert D_T >= D_S, "New width must be greater than or equal to old width"

    # Compute the number of times to repeat x
    num_repeats = D_T // D_S

    # Compute the number of zeros to append
    num_zeros = D_T % D_S

    # Concatenate the repeated vector and zeros
    y = torch.cat([x.repeat(num_repeats), torch.zeros(num_zeros)])

    return y


def vector_circular_expansion(x, D_S, D_T, alpha=1, beta=1):
    """
    Computes the vector circular expansion of a given input vector `x` of dimension `D_S`
    to a new dimension `D_T`.

    The vector circular expansion is defined as:

    xcirc = Vcirc(x) = Concat[x, ..., x, x[: Dr mod Ds]] ∈ ℝ^Dr

    where:
    - x ∈ RDs is the input vector of dimension Ds
    - xcirc is the circular expanded vector of dimension Dr with Dr ≥ Ds
    - [Dr/Ds] is the number of repeated copies of x
    - Dr mod Ds is the number of elements from the beginning of x to prepend

    Args:
        x (torch.Tensor(D_S)): The input vector of dimension `D_S`.
        D_S (int): The dimension of the input vector `x`.
        D_T (int): The desired dimension of the output vector.

    Returns:
        torch.Tensor(D_T): The circular expanded vector of dimension `D_T`.
    """

    assert D_T >= D_S, "New width must be greater than or equal to old width"

    # Compute the number of times to repeat x
    num_repeats = D_T // D_S

    # Compute the number of elements to prepend from the beginning of x
    num_prepend = D_T % D_S

    # Concatenate the repeated vector and prepended elements
    y = torch.cat([x[:num_prepend]*alpha, x[num_prepend:], x.repeat(num_repeats-1), x[:num_prepend]*beta])

    return y


def vector_random_expansion(x, D_S, D_T, random_param='uniform'):
    """
    Computes the vector random expansion of a given input vector `x` of dimension `D_S`
    to a new dimension `D_T`.

    The vector random expansion is defined as:

    xrand = Vrand(x) = Concat[x^T, ..., x^T, CT ∈ ℝ^Dr]

    where:
    - x ∈ RDs is the input vector of dimension Ds
    - xrand is the random expanded vector of dimension Dr with Dr ≥ Ds
    - [Dr/Ds] is the number of repeated copies of x^T
    - CT ∈ ℝ^Dr is an arbitrary vector of dimension Dr mod Ds

    Args:
        x (torch.Tensor(D_S)): The input vector of dimension `D_S`.
        D_S (int): The dimension of the input vector `x`.
        D_T (int): The desired dimension of the output vector.

    Returns:
        torch.Tensor(D_T): The random expanded vector of dimension `D_T`.
    """

    assert D_T >= D_S, "New width must be greater than or equal to old width"

    # Compute the number of times to repeat x
    num_repeats = D_T // D_S

    # Compute the number of elements to pad in the random vector
    num_pad = D_T % D_S

    # Generate a random vector of dimension num_pad
    if random_param == 'uniform':
        random_pad = torch.rand(num_pad)
    elif random_param == 'normal':
        random_pad = torch.randn(num_pad) * 0.02

    # Concatenate the repeated vector and random vector
    y = torch.cat([x.repeat(num_repeats), random_pad])

    return y


#########################
# matrix row expansions #
#########################

def matrix_row_average_expansion(x, D_S, D_T):
    """
    Computes the row-average expansion of a matrix `x` of dimension (Ds x P) to a new number of rows (D_T).

    The row-average expansion is defined as:

    Mrow,avg = Erow,avg(M) = Concat[MT, ..., MT, m, ..., m] ∈ ℝDT x P

    where:
    - M ∈ RDS x P is the input matrix of dimension Ds x P.
    - Mrow,avg ∈ ℝDT x P is the row-average expanded matrix of dimension DT x P with DT >= Ds.
    - MT ∈ ℝP x Ds is the transpose of the input matrix.
    - m ∈ ℝ1 x P is the row-wise average of the input matrix M.
    - [DT/Ds] is the number of repeated copies of MT.
    - DT mod Ds is the number of copies of the row-wise average vector m appended.

    Args:
        x (torch.Tensor): The input matrix of dimension (D_S x P).
        D_S (int): The old number of rows in the input matrix.
        D_T (int): The desired number of rows in the output matrix.

    Returns:
        torch.Tensor: The row-average expanded matrix of dimension (D_T x P).
    """

    assert D_T >= D_S, "New number of rows must be greater than or equal to old number of rows"

    # Compute the row-wise mean of the input matrix
    row_mean = torch.mean(x, dim=0, keepdim=True)

    # Compute the number of times to append the row mean vector
    num_append = D_T % D_S
    num_repeats = D_T // D_S

    # Concatenate the row mean vectors to the [x[0,:].T, ..., x[-1,:].T] matrix
    y = torch.cat([x.repeat(num_repeats, 1), row_mean.repeat(num_append, 1)])

    return y


def matrix_row_zero_expansion(x, D_S, D_T):
    """
    Computes the row-zero expansion of a matrix `x` of dimension (Ds x P) to a new number of rows (D_T).

    The row-zero expansion is defined as:

    Mrow,zero = Erow,zero(M) = Concat[MT, ..., MT, 0, ..., 0] ∈ ℝDT x P

    where:
    - M ∈ RDS x P is the input matrix of dimension Ds x P.
    - Mrow,avg ∈ ℝDT x P is the row-average expanded matrix of dimension DT x P with DT >= Ds.
    - MT ∈ ℝP x Ds is the transpose of the input matrix.
    - m ∈ ℝ1 x P is the row-wise average of the input matrix M.
    - [DT/Ds] is the number of repeated copies of MT.
    - DT mod Ds is the number of copies of the row-wise average vector m appended.

    Args:
        x (torch.Tensor): The input matrix of dimension (D_S x P).
        D_S (int): The old number of rows in the input matrix.
        D_T (int): The desired number of rows in the output matrix.

    Returns:
        torch.Tensor: The row-average expanded matrix of dimension (D_T x P).
    """

    assert D_T >= D_S, "New number of rows must be greater than or equal to old number of rows"

    P = x.shape[1]

    zeros = torch.zeros(1, P)

    # Compute the number of times to append the row mean vector
    num_append = D_T % D_S
    num_repeats = D_T // D_S

    # Concatenate the zero vectors to the [x[0,:].T, ..., x[-1,:].T] matrix
    y = torch.cat([x.repeat(num_repeats, 1), zeros.repeat(num_append, 1)])

    return y


def matrix_row_circular_expansion(x, D_S, D_T, alpha=1, beta=1):
    """
    Computes the row-circular expansion of a matrix `x` of dimension (Ds x P) to a new number of rows (D_T).

    The row-circular expansion is defined as:

    Mrow,circ = Erow,circ(M) = Concat[M, ..., M, M[: DT mod Ds, :]] ∈ ℝDT x P

    where:
    - M ∈ RDS x P is the input matrix of dimension Ds x P.
    - Mrow,circ ∈ ℝDT x P is the row-circular expanded matrix of dimension DT x P with DT >= Ds.
    - [DT/Ds] is the number of repeated copies of M.
    - DT mod Ds is the number of rows from the beginning of M to prepend.

    Args:
        x (torch.Tensor): The input matrix of dimension (D_S x P).
        D_S (int): The old number of rows in the input matrix.
        D_T (int): The desired number of rows in the output matrix.

    Returns:
        torch.Tensor: The row-circular expanded matrix of dimension (D_T x P).
    """

    assert D_T >= D_S, "New number of rows must be greater than or equal to old number of rows"

    # Compute the number of times to prepend the input matrix
    num_prepend = D_T % D_S
    num_repeats = D_T // D_S

    # Concatenate the input matrix to the [x[0,:], ..., x[-1,:]] matrix
    y = torch.cat([x[:num_prepend, :]*alpha, x[num_prepend:, :], x.repeat(num_repeats-1, 1), x[:num_prepend, :]*beta])

    return y


def matrix_row_random_expansion(x, D_S, D_T):
    """
    Computes the row-random expansion of a matrix `x` of dimension (Ds x P) to a new number of rows (D_T).

    The row-random expansion is defined as:

    Mrow,rand = Erow,rand(M) = Concat[M, ..., M, C] ∈ ℝDT x P

    where:
    - M ∈ RDS x P is the input matrix of dimension Ds x P.
    - Mrow,rand ∈ ℝDT x P is the row-random expanded matrix of dimension DT x P with DT >= Ds.
    - [DT/Ds] is the number of repeated copies of M.
    - C ∈ ℝDT mod Ds x P is an arbitrary matrix of dimension DT mod Ds x P.

    Args:
        x (torch.Tensor): The input matrix of dimension (D_S x P).
        D_S (int): The old number of rows in the input matrix.
        D_T (int): The desired number of rows in the output matrix.

    Returns:
        torch.Tensor: The row-random expanded matrix of dimension (D_T x P).
    """

    assert D_T >= D_S, "New number of rows must be greater than or equal to old number of rows"

    # Compute the number of times to prepend the input matrix
    num_pad = D_T % D_S
    num_repeats = D_T // D_S

    # Generate a random matrix of dimension (num_pad, P)
    random_pad = torch.rand(num_pad, x.shape[1])

    # Concatenate the input matrix to the random matrix
    y = torch.cat([x.repeat(num_repeats, 1), random_pad])

    return y


############################
# matrix column expansions #
############################

def matrix_column_average_expansion(x, D_S, D_T, alpha=1, beta=1):
    """
    Computes the column-average expansion of a matrix `x` of dimension (D_S x P) to a new number of columns (D_T).

    The column-average expansion is defined as:

    Mcol,avg = Ecol,avg(M) = Concat[M, ..., M, m, ..., m] ∈ ℝD_S x D_T

    where:
    - M ∈ RDS x P is the input matrix of dimension Ds x P.
    - Mcol,avg ∈ ℝDS x DT is the column-average expanded matrix of dimension DS x DT with DT >= P.
    - m ∈ ℝDS x 1 is the column-wise average of the input matrix M.
    - [DT/P] is the number of repeated copies of M.
    - DT mod P is the number of copies of the column-wise average vector m appended.

    Args:
        x (torch.Tensor): The input matrix of dimension (D_S x P).
        D_S (int): The old number of columns in the input matrix.
        D_T (int): The desired number of columns in the output matrix.

    Returns:
        torch.Tensor: The column-average expanded matrix of dimension (D_S x D_T).
    """

    assert D_T >= D_S, "New number of columns must be greater than or equal to old number of columns"

    # Compute the column-wise mean of the input matrix
    col_mean = torch.mean(x, dim=1, keepdim=True)

    # Compute the number of times to append the column mean vector
    num_append = D_T % D_S
    num_repeats = D_T // D_S

    # Concatenate the column mean vectors to the [x[:, 0], ..., x[:, -1]] matrix
    y = torch.cat([x*alpha, x.repeat(1, num_repeats-1), col_mean.repeat(1, num_append)*beta], dim=1)

    return y


def matrix_column_zero_expansion(x, D_S, D_T):
    """
    Computes the column-zero expansion of a matrix `x` of dimension (D_S x P) to a new number of columns (D_T).

    The column-zero expansion is defined as:

    Mcol,zero = Ecol,zero(M) = Concat[M, ..., M, 0, ..., 0] ∈ ℝD_S x D_T

    where:
    - M ∈ RDS x P is the input matrix of dimension Ds x P.
    - Mcol,zero ∈ ℝDS x DT is the column-zero expanded matrix of dimension DS x DT with DT >= P.
    - [DT/P] is the number of repeated copies of M.
    - DT mod P is the number of copies of the zero vector appended.

    Args:
        x (torch.Tensor): The input matrix of dimension (P x D_S).
        D_S (int): The old number of columns in the input matrix.
        D_T (int): The desired number of columns in the output matrix.

    Returns:
        torch.Tensor: The column-zero expanded matrix of dimension (D_S x D_T).
    """

    assert D_T >= D_S, "New number of columns must be greater than or equal to old number of columns"

    P = x.shape[0]

    zeros = torch.zeros(P, 1)

    # Compute the number of times to append the zero vector
    num_append = D_T % D_S
    num_repeats = D_T // D_S

    # Concatenate the zero vectors to the [x[:, 0], ..., x[:, -1]] matrix
    y = torch.cat([x.repeat(1, num_repeats), zeros.repeat(1, num_append)], dim=1)

    return y


def matrix_column_circular_expansion(x, D_S, D_T, alpha=1, beta=1):
    """
    Computes the column-circular expansion of a matrix `x` of dimension (D_S x P) to a new number of columns (D_T).

    The column-circular expansion is defined as:

    Mcol,circ = Ecol,circ(M) = Concat[M, ..., M, M[:, : DT mod P]] ∈ ℝD_S x D_T

    where:
    - M ∈ RDS x P is the input matrix of dimension Ds x P.
    - Mcol,circ ∈ ℝDS x DT is the column-circular expanded matrix of dimension DS x DT with DT >= P.
    - [DT/P] is the number of repeated copies of M.
    - DT mod P is the number of columns from the beginning of M to prepend.

    Args:
        x (torch.Tensor): The input matrix of dimension (D_S x P).
        D_S (int): The old number of columns in the input matrix.
        D_T (int): The desired number of columns in the output matrix.

    Returns:
        torch.Tensor: The column-circular expanded matrix of dimension (D_S x D_T).
    """

    assert D_T >= D_S, "New number of columns must be greater than or equal to old number of columns"

    # Compute the number of times to prepend the input matrix
    num_prepend = D_T % D_S
    num_repeats = D_T // D_S

    # Concatenate the input matrix to the [x[:, 0], ..., x[:, -1]] matrix
    y = torch.cat([x[:, :num_prepend]*alpha, x[:, num_prepend:], x.repeat(1, num_repeats-1), x[:, :num_prepend]*beta], dim=1)

    return y


def matrix_column_random_expansion(x, D_S, D_T, random_param='normal', std_dev=0.02):
    """
    Computes the column-random expansion of a matrix `x` of dimension (D_S x P) to a new number of columns (D_T).

    The column-random expansion is defined as:

    Mcol,rand = Ecol,rand(M) = Concat[M, ..., M, C] ∈ ℝD_S x D_T

    where:
    - M ∈ RDS x P is the input matrix of dimension Ds x P.
    - Mcol,rand ∈ ℝDS x DT is the column-random expanded matrix of dimension DS x DT with DT >= P.
    - [DT/P] is the number of repeated copies of M.
    - C ∈ ℝDS x DT mod P is an arbitrary matrix of dimension DS x DT mod P.

    Args:
        x (torch.Tensor): The input matrix of dimension (P x D_S).
        D_S (int): The old number of columns in the input matrix.
        D_T (int): The desired number of columns in the output matrix.
        random_param (str): The random parameter {'uniform', 'normal'} to use. Default is 'normal'.
        std_dev (float): The standard deviation of the normal distribution. Default is 0.02.
        
    Returns:
        torch.Tensor: The column-random expanded matrix of dimension (D_S x D_T).
    """

    assert D_T >= D_S, "New number of columns must be greater than or equal to old number of columns"

    # Compute the number of times to prepend the input matrix
    num_pad = D_T % D_S
    num_repeats = D_T // D_S

    if random_param == 'uniform':
        # Generate a random matrix of dimension (P, num_pad) from
        # uniform distribution Unif(-1, 1)
        random_pad = torch.rand(x.shape[0], num_pad) * 2 - 1
    elif random_param == 'normal':
        # Generate a random matrix of dimension (P, num_pad) from
        # normal distribution N(0, 0.02^2)
        random_pad = torch.randn(x.shape[0], num_pad) * std_dev

    # Concatenate the input matrix to the random matrix
    y = torch.cat([x.repeat(1, num_repeats), random_pad], dim=1)

    return y


def wide_param(name, weight, old_width, new_width):

    if 'embed_in' in name:
        # For input embeddings, the columns are average expanded
        D_S = weight.shape[1]
        D_T = int(D_S * new_width/old_width)
        # alpha = (2 - D_T/D_S)
        # return matrix_column_circular_expansion(weight, D_S, D_T, alpha=0.5, beta=0.5)
        return matrix_column_zero_expansion(weight, D_S, D_T) # (V_Id, V_Zero)
        return matrix_column_average_expansion(weight, D_S, D_T, alpha=1, beta=0)
    
    elif 'embed_out' in name:
        # For weight, column expanded by circular expansion
        if "weight" in name:
            D_S = weight.shape[1]
            D_T = int(D_S * new_width/old_width)
            # weight = matrix_column_circular_expansion(weight, D_S, D_T, alpha=1, beta=1)
            weight = matrix_column_zero_expansion(weight, D_S, D_T)

        elif "bias" in name:
            return weight
    
    elif "input_layernorm" in name or 'post_attention_layernorm' in name or 'final_layer_norm' in name:
        # For layernorm, weight is expanded as \eta*vector_randdom_expansion() where 
        #   \eta is floor(D_T/D_S)*(D_S/D_T) and 
        #   bias is expanded as vector_zero_expansion()
        D_S = weight.shape[0]
        D_T = int(D_S * new_width/old_width)

        eta = torch.sqrt(torch.tensor(D_S/D_T)).item()
        if "weight" in name:
            return eta * vector_random_expansion(weight, D_S, D_T, random_param='uniform')
        elif "bias" in name:
            return vector_zero_expansion(weight, D_S, D_T)
        
    # elif "final_layer_norm" in name:
    #     D_S = weight.shape[0]
    #     D_T = int(D_S * new_width/old_width)

    #     return vector_zero_expansion(weight, D_S, D_T)

    
    elif "attention" in name and "query_key_value" in name:
        # For attention qkv, the dimensions of W^K, W^Q, W^V are expanded from
        #  (d_k, D_S) to (d_k, D_T) using matrix_column_random_expansion()
        

        if "weight" in name:
            # TODO: fix
            D_S = weight.shape[1]
            D_T = int(D_S * new_width/old_width)
            weight = matrix_column_random_expansion(weight, D_S, D_T, random_param='normal', std_dev=0.002)
            D_S = weight.shape[0]
            D_T = int(D_S * new_width/old_width)
            return matrix_row_circular_expansion(weight, D_S, D_T)
        
        elif "bias" in name:
            D_S = weight.shape[0]
            D_T = int(D_S * new_width/old_width)
            return vector_circular_expansion(weight, D_S, D_T)
        
    elif "attention" in name and "dense" in name:
        # For attention projection matrices, the weight is expanded from (D_S, D_S)
        # to (D_T, D_T) using matrix_row_average_expansion() followed by 
        # matrix_column_circular_expansion()
        # the bias is expanded from (D_S) to (D_T) using 
        # vector_average_expansion()

        if "weight" in name:
            D_S = weight.shape[0]
            D_T = int(D_S * new_width/old_width)
            weight = matrix_row_average_expansion(weight, D_S, D_T)
            D_S = weight.shape[1]
            D_T = int(D_S * new_width/old_width)
            return matrix_column_circular_expansion(weight, D_S, D_T)
        
        elif "bias" in name:
            D_S = weight.shape[0]
            D_T = int(D_S * new_width/old_width)
            return vector_average_expansion(weight, D_S, D_T)

    elif "mlp" in name and "dense_h_to_4h" in name:
        # (V_zero, V_circ)

        # For the weight, first we expand rows by circular expansion and
        # columns by random expansion. For the bias, we expand by circular expansion

        if "weight" in name:
            D_S = weight.shape[0]
            D_T = int(D_S * new_width/old_width)
            weight = matrix_row_circular_expansion(weight, D_S, D_T, alpha=1, beta=1)
            D_S = weight.shape[1]
            D_T = int(D_S * new_width/old_width)
            return matrix_column_random_expansion(weight, D_S, D_T, random_param='normal', std_dev=0.002)
        
        elif "bias" in name:
            D_S = weight.shape[0]
            D_T = int(D_S * new_width/old_width)
            return vector_circular_expansion(weight, D_S, D_T, alpha=1, beta=1)
        
    elif "mlp" in name and "dense_4h_to_h" in name:
        # 
        # For the weight, first we expand rows by average expansion and
        # columns by circular expansion. For the bias, we expand by average 
        # expansion

        if "weight" in name:
            D_S = weight.shape[0]
            D_T = int(D_S * new_width/old_width)
            # weight = matrix_row_zero_expansion(weight, D_S, D_T)
            weight = matrix_row_average_expansion(weight, D_S, D_T)
            D_S = weight.shape[1]
            D_T = int(D_S * new_width/old_width)
            return matrix_column_circular_expansion(weight, D_S, D_T, alpha=0.5, beta=0.5)
        
        elif "bias" in name:
            D_S = weight.shape[0]
            D_T = int(D_S * new_width/old_width)
            # return vector_zero_expansion(weight, D_S, D_T)
            return vector_average_expansion(weight, D_S, D_T)

    return weight

def wide_state_dict(old_state_dict, old_width, new_width):
    """
    Function preserving expansion of a state dict from size `old_width` to `new_width`.

    Args:
        old_state_dict (dict): The state dict of the model to expand
        old_width (int): The old width of the model
        new_width (int): The new width of the model

    Returns:
        dict: The expanded state dict
    """

    # Create new state dict
    new_state_dict = {}
    for key, weight in old_state_dict.items():
        # print(f'key: {key}, shape: {weight.shape}')
        new_state_dict[key] = wide_param(key, weight, old_width, new_width)
        # print(f'key: {key}, new shape: {new_state_dict[key].shape}')
        # print('-'*50)
    return new_state_dict



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

    # Reproducibility
    torch.manual_seed(0)


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

    # expand_width(model_70m, 512, 768)

    model_70m_wide = expand_width(model_70m, 512, 768)
    print(f"Expanded model: {model_70m_wide.config.hidden_size}")
    inputs = tokenizer_70m("Finish the following sentence:\nRaindrops on roses", return_tensors="pt")

    print(f'hidden state: {model_70m_wide(**inputs)[0].shape}')

    tokens = model_70m_wide.generate(**inputs)
    print(tokenizer_70m.decode(tokens[0]))
