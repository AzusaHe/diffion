"""Replay a fixed-shape loss/gradient calculation without repeated Python dispatch."""
import os

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from pinn.material import RHO_COEFF, material as evaluate_material

_density_coefficients = {}


def prepare_material(device, dtype):
    coeff = torch.tensor(RHO_COEFF, dtype=dtype, device=device)
    _density_coefficients[coeff.device, dtype] = coeff


def material(concentration, theta):
    coeff = _density_coefficients.get((concentration.device, concentration.dtype), RHO_COEFF)
    return evaluate_material(concentration, theta, coeff)


def replayable(function, device):
    """Copy each new argument into stable graph storage and return independent results."""
    if str(device) != "cuda" or os.environ.get("DIFFION_PINN_GRAPHS", "0") != "1":
        return function
    graph = None
    prev_spec = None
    targets = None
    output_leaves = None
    output_spec = None
    def evaluate(*args):
        nonlocal graph, prev_spec, targets, output_leaves, output_spec
        leaves, spec = tree_flatten(args)
        if graph is None or spec != prev_spec:
            targets = [leaf.detach().clone() for leaf in leaves]
            static = tree_unflatten(targets, spec)
            # Warm allocator and cuBLAS on the capture stream before capturing.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                function(*static)
                function(*static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                output_leaves, output_spec = tree_flatten(function(*static))
            prev_spec = spec
        torch._foreach_copy_(targets, leaves)
        graph.replay()
        # L-BFGS retains its previous objective, gradient and loss components.
        return tree_unflatten([x.clone() for x in output_leaves], output_spec)
    evaluate.eager = function
    return evaluate


def graph_adam(params, n, rates, multipliers, step_fn, draw, ctr, log_every, record, attach, max_wall_s):
    """Capture the complete Adam update; feed the original RNG samples and cosine schedule each step."""
    leaves, spec = tree_flatten(params)
    static = [p.clone() for p in leaves]
    current = tree_unflatten(static, spec)
    m = [torch.zeros_like(p) for p in leaves]
    v = [torch.zeros_like(p) for p in leaves]
    coef = torch.empty(3, device=static[0].device, dtype=static[0].dtype)
    schedule = torch.tensor([[1.0 - 0.9 ** (i + 1), 1.0 - 0.999 ** (i + 1), rate]
                             for i, rate in enumerate(rates)], dtype=coef.dtype, device=coef.device)
    # Allocate storage from an actual draw without advancing the generator twice on the first iteration.
    x, t = draw()
    f = step_fn.eager
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        f(current, x, t)
        f(current, x, t)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        grads, (_, (loss, parts)) = f(current, x, t)
        gs = tree_flatten(grads)[0]
        with torch.no_grad():
            for j, pj in enumerate(static):
                m[j].mul_(0.9).add_(gs[j], alpha=1.0-0.9)
                v[j].mul_(0.999).addcmul_(gs[j], gs[j], value=1.0-0.999)
                pj.sub_(coef[2] * multipliers[j] * (m[j] / coef[0]) /
                        (torch.sqrt(v[j] / coef[1]) + 1e-8))
    torch._foreach_copy_(static, leaves)
    torch._foreach_zero_(m)
    torch._foreach_zero_(v)
    for i in range(n):
        logged = ctr["step"] % log_every == 0
        if logged and record(current) > max_wall_s:
            return current, True
        if i:
            xx, tt = draw()
            x.copy_(xx)
            t.copy_(tt)
        coef.copy_(schedule[i])
        graph.replay()
        ctr["evals"] += 1
        if logged:
            attach(loss, parts)
        ctr["step"] += 1
    return current, False
