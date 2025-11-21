# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import grouped_gemm_backend as backend

# # register them intro graphs
# torch.compiler.allow_in_graph(backend.gmm_sm89)
# torch.compiler.allow_in_graph(backend.gmm)
@torch.library.custom_op("grouped_gemm_backend::gmm", mutates_args={"c"}, device_types="cuda")
def gmm_handler(a : torch.Tensor, 
                b : torch.Tensor, 
                c : torch.Tensor, 
                batch_sizes : torch.Tensor, 
                trans_a : bool, 
                trans_b : bool) -> torch.Tensor:
    return backend.gmm(a, b, c, batch_sizes, trans_a, trans_b)

@torch.library.custom_op("grouped_gemm_backend::gmm_sm89", mutates_args={"c"}, device_types="cuda")
def gmm_sm89_handler(a : torch.Tensor, 
                b : torch.Tensor, 
                c : torch.Tensor, 
                batch_sizes : torch.Tensor, 
                trans_a : bool, 
                trans_b : bool) -> torch.Tensor:
    return backend.gmm_sm89(a, b, c, batch_sizes, trans_a, trans_b)

def _(a : torch.Tensor, 
        b : torch.Tensor, 
        c : torch.Tensor, 
        batch_sizes : torch.Tensor, 
        trans_a : bool, 
        trans_b : bool) -> torch.Tensor:
    assert a.is_contiguous()
    assert b.is_contiguous()
    assert c.is_contiguous()
    if not trans_a and trans_b:
        # forward pass
        assert a.ndim == 2
        assert b.ndim == 3
        assert a.shape[-1] == b.shape[-1]
        return torch.empty(a.shape[0], b.shape[1])
    elif trans_a and not trans_b:
        # a : [num_tokens, hidden_out]
        # b : [num_tokens, hidden_in]
        # c : [num_experts, hidden_out, hidden_in]
        assert a.ndim == 2
        assert b.ndim == 2
        assert a.shape[0] == b.shape[0]
        return torch.empty(batch_sizes.shape[0], a.shape[-1], b.shape[-1])
    elif not trans_a and not trans_b:
        # a : [num_tokens, hidden_out]
        # b : [num_expert, hidden_out, hidden_in]
        # c : [num_tokens, hidden_in]
        assert a.ndim == 2
        assert b.ndim == 3
        assert a.shape[1] == b.shape[1]
        return torch.empty(a.shape[0], b.shape[-1])
    else:
        assert False

gmm_handler.register_fake(_)
gmm_sm89_handler.register_fake(_)

def _allocate_output(a, b, batch_sizes, trans_a, trans_b):
    assert not (trans_a and trans_b)
    assert batch_sizes.ndim == 1, "Expected 1d tensor for batch_sizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if trans_a else 3)

    shape = (
        (batch_sizes.shape[0], a.shape[1], b.shape[1])
        if trans_a else
        (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    )
    return torch.empty(*shape, device=a.device, dtype=a.dtype)

# Assuming 'backend' and '_allocate_output' are available

def gmm(a : torch.Tensor, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    gmm_func = torch.ops.grouped_gemm_backend.gmm

    if a.is_cuda:
        major, minor = torch.cuda.get_device_capability(a.get_device())
        
        if major == 8 and minor == 9:
            gmm_func = torch.ops.grouped_gemm_backend.gmm_sm89

    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    
    gmm_func(a, b, c, batch_sizes, trans_a, trans_b)
    
    return c
