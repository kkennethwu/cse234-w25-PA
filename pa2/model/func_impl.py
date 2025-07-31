import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):
    """
    Prepare necessary information for later communications in forward and backward passes.

    Parameters
    ----------
    comm : Communicator
        The global MPI communicator.
    rank : int
        The global rank of the process.
    mp_size : int
        Model Parallel size.
    dp_size : int
        Data Parallel size.
    fc_layer : str
        Identifier for the fully-connected layer. It must be one of:
        'fc_q', 'fc_k', 'fc_v', or 'fc_o'.
        - For 'fc_q', 'fc_k', and 'fc_v', the partitioning is along the output dimension.
        - For 'fc_o', the partitioning is along the input dimension.
    in_dim : int
        Original input feature dimension.
    out_dim : int
        Original output feature dimension.

    Returns
    -------
    mp_idx : int
        Model parallel index (position within a data parallel replica).
    dp_idx : int
        Data parallel index (which replica this process belongs to).
    mp_comm : Communicator
        The model parallel communicator (all processes in one data parallel replica).
    dp_comm : Communicator
        The data parallel communicator (all processes holding the same weight shard).
    part_in_dim : int
        The partitioned input dimension for the FC layer.
    part_out_dim : int
        The partitioned output dimension for the FC layer.
    """
    #TODO: Your code here

    mp_idx = rank % mp_size  # Model parallel index
    dp_idx = rank // mp_size  # Data parallel index
    
    color_mp = dp_idx  # Color for model parallel communicator
    mp_comm = comm.Split(color=color_mp, key=mp_idx)
    
    color_dp = mp_idx
    dp_comm = comm.Split(color=color_dp, key=dp_idx)
    
    # X (input): [batch_size, seq_length, in_dim] 
    # W_q, W_k, W_v (weights): [in_dim, in_dim]
    # Q = X @ W_q, K = X @ W_k, V = X @ W_v: [batch_size, seq_length, H, head_dim] (in_dim = H * head_dim for multi-head attention)
    # scores = Q @ K.transpose(-1, -2) / sqrt(D_head)  # [batch_size, H, seq_length, seq_length]
    # A = softmax(scores, dim=-1) # [batch_size, H, seq_length, seq_length]
    # Z = A @ V  # [batch_size, H, seq_length, head_dim] --transpose&reshape--> [batch_size, seq_length, in_dim]
    # W_o (weight): [in_dim, out_dim]
    # O = Z @ W_o: [batch_size, seq_length, out_dim]

    if fc_layer in ["fc_q", "fc_k", "fc_v"]:
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size # for broadcast
    elif fc_layer == "fc_o":
        part_in_dim = in_dim // mp_size # for all-reduce
        part_out_dim = out_dim
    
    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim

def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward inputs from all model-parallel nodes.

    Each node holds a piece of the full input with shape:
      (batch_size, seq_length, part_in_dim)
    After gathering, the full input should have shape:
      (batch_size, seq_length, part_in_dim * mp_size)
    """
    #TODO: Your code here
    x = np.ascontiguousarray(x)
    collected_x = np.empty((mp_size, x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype)
    mp_comm.Allgather([x, MPI.FLOAT], [collected_x, MPI.FLOAT])
    collected_x = np.transpose(collected_x, (1, 2, 0, 3))
    
    # 合併最後兩個維度：(batch, seq, mp_size, part_dim) -> (batch, seq, mp_size * part_dim)
    collected_x = collected_x.reshape(x.shape[0], x.shape[1], -1)

    return collected_x


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward outputs from all model-parallel nodes.

    Each node holds a piece of the full output with shape:
      (batch_size, seq_length, part_out_dim)
    After gathering, the full output should have shape:
      (batch_size, seq_length, part_out_dim * mp_size)
    """
    #TODO: Your code here
    out = np.ascontiguousarray(out) # to solve: FAILED tests/test_transformer_forward.py::test_fc_o_naive_mp_forward_x_3d - BufferError: dlpack: buffer is not contiguous
    collected_out = np.empty((mp_size, out.shape[0], out.shape[1], out.shape[2]), dtype=out.dtype)
    mp_comm.Allgather([out, MPI.FLOAT], [collected_out, MPI.FLOAT])
    # 重新排列維度並合併
    collected_out = np.transpose(collected_out, (1, 2, 0, 3))
    collected_out = collected_out.reshape(out.shape[0], out.shape[1], -1)

    return collected_out

def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Collect the fc output layer's output gradient for the local MP node.
    
    In our setup, the full output_grad is a 3-D tensor of shape 
        (batch_size, seq_length, out_dim),
    and the fully connected layer's weight is partitioned along out_dim.
    Therefore, we split output_grad along axis=2 into mp_size parts and
    return the part corresponding to mp_group_idx.
    
    Parameters
    ----------
    output_grad : np.ndarray
        The full output gradient from fc_o with shape 
        (batch_size, seq_length, out_dim).
    mp_group_idx : int
        The current model parallel node's index.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_output_grad : np.ndarray
        The local output gradient for this MP node with shape 
        (batch_size, seq_length, out_dim // mp_size).
    """
    #TODO: Your code here
    part_out_dim = output_grad.shape[2] // mp_size
    start_idx = mp_group_idx * part_out_dim
    end_idx = start_idx + part_out_dim
    collected_output_grad = output_grad[:, :, start_idx:end_idx]
    return collected_output_grad
    


def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Use reduce-scatter / all-to-all to combine the contributions for grad_x from all nodes
    and scatter the reduced result along the input feature dimension.
    
    The grad_x tensor (gradient with respect to fc_o's input) has shape
        (batch_size, seq_length, in_dim),
    and the fc_o's weight matrix is sharded along the in_dim axis. In the 
    backward pass, each node computes a local grad_x and then these must be 
    summed across nodes. Instead of summing the full tensor and then slicing,
    we perform a reduce-scatter / all-to-all.
    
    Parameters
    ----------
    grad_x : np.ndarray
        The locally computed grad_x for fc_o, of shape 
        (batch_size, seq_length, in_dim).
    mp_comm :
        The model parallel communicator. It is assumed to expose methods such as reduce-scatter / all-to-all.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_grad_x : np.ndarray
        The reduced and scattered grad_x with shape 
        (batch_size, seq_length, in_dim // mp_size).
    """
    #TODO: Your code here
    part_in_dim = grad_x.shape[2] // mp_size
    collected_grad_x = np.empty((grad_x.shape[0], grad_x.shape[1], part_in_dim), dtype=grad_x.dtype)
    grad_x_reshaped = grad_x.reshape(grad_x.shape[0], grad_x.shape[1], mp_size, part_in_dim)
    grad_x_reshaped_transpose = np.transpose(grad_x_reshaped, (2, 0, 1, 3))  # (mp_size, batch_size, seq_length, part_in_dim)
    grad_x_reshaped_transpose = np.ascontiguousarray(grad_x_reshaped_transpose)  # Ensure the array is contiguous
    mp_comm.Reduce_scatter(grad_x_reshaped_transpose, collected_grad_x, op=MPI.SUM)

    return collected_grad_x
