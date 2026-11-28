"""Reusable CUDA Adam updates for independent single-material CV PINN fits."""
from collections import OrderedDict
from time import perf_counter

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from .gpu_eval import _density_coefficients


class _ResidualSamples:
    """Use the same alternating x/u Philox calls as the original residual sampler."""

    def __init__(self, generator, samples, steps, points):
        rng = torch.Generator(device=generator.device)
        rng.set_state(generator.get_state())
        currents = samples["iotas"]
        shape = (1, currents.shape[1], points)
        options = dict(generator=rng, dtype=currents.dtype, device=currents.device)
        x, u = [], []
        for _ in range(steps):
            x.append(torch.rand(shape, **options))
            u.append(torch.rand(shape, **options))
        self.x = torch.stack(x)
        self.u = torch.stack(u)
        self.end_state = rng.get_state()

    def times(self, samples):
        lo = samples["s_lo"][:, None, None]
        return samples["end_times"][:, :, None] * (lo + (1 - lo) * self.u) ** 2


class _AdamUpdate:
    def __init__(self, params, function, draws, rates, multipliers, observations):
        leaves, self.spec = tree_flatten(params)
        self.weights = [p.clone() for p in leaves]
        self.current = tree_unflatten(self.weights, self.spec)
        self.moments = [torch.zeros_like(p) for p in leaves]
        self.squares = [torch.zeros_like(p) for p in leaves]
        # Graph replay uses original storage addresses. Keep the closure (including
        # quadrature tensors), material constants and observation tensors alive.
        self.function = function
        self.material_constants = tuple(_density_coefficients.values())
        self.observations = observations
        self.x = draws.x
        self.t = draws.times(observations)
        self.index = torch.zeros(1, device=self.x.device, dtype=torch.int64)
        self.schedule = leaves[0].new_tensor([
            (1.0 - 0.9 ** (i + 1), 1.0 - 0.999 ** (i + 1), lr) for i, lr in enumerate(rates)
        ])
        side = torch.cuda.Stream(device=self.x.device)
        side.wait_stream(torch.cuda.current_stream(self.x.device))
        with torch.cuda.stream(side):
            function(self.current, self.x[0], self.t[0])
            function(self.current, self.x[0], self.t[0])
        torch.cuda.current_stream(self.x.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            x = self.x.index_select(0, self.index)[0]
            t = self.t.index_select(0, self.index)[0]
            coefficient = self.schedule.index_select(0, self.index)[0]
            gradients, (_, (self.loss, self.parts)) = function(self.current, x, t)
            with torch.no_grad():
                for p, grad, m, v, mult in zip(self.weights, tree_flatten(gradients)[0],
                                               self.moments, self.squares, multipliers):
                    m.mul_(0.9).add_(grad, alpha=1.0 - 0.9)
                    v.mul_(0.999).addcmul_(grad, grad, value=1.0 - 0.999)
                    p.sub_(coefficient[2] * mult * (m / coefficient[0]) /
                           (torch.sqrt(v / coefficient[1]) + 1e-8))
                self.index.add_(1)

    def reset(self, params, observations, draws):
        if self.observations is not observations:
            for key, value in observations.items():
                self.observations[key].copy_(value)
        self.t.copy_(draws.times(observations))
        self.index.zero_()
        torch._foreach_copy_(self.weights, tree_flatten(params)[0])
        torch._foreach_zero_(self.moments)
        torch._foreach_zero_(self.squares)

    def independent_params(self):
        return tree_unflatten([p.clone() for p in self.weights], self.spec)


class SingleFitGraphCache:
    """Share graph setup across cases with identical shapes and Adam settings.

    All valid native observations still enter each update. Only one observation
    shape and up to five optimizer graphs are retained at a time.
    """
    def __init__(self):
        self._shape = None
        self._graphs = OrderedDict()
        self._draws = OrderedDict()
        self._counts = dict(hits=0, misses=0, capture_s=0.0, rng_setup_s=0.0)

    def stats(self):
        return self._counts.copy()

    def train(self, params, rates, multipliers, function, loss, observations, generator, stage,
              counter, log_every, record, attach, snapshot_steps, snapshot):
        steps = len(rates)
        currents = observations["iotas"]
        layout = tuple((k, v.shape, v.stride(), v.dtype, v.device) for k, v in observations.items())
        if layout != self._shape:
            self._graphs.clear()
            self._shape = layout
        stream = torch.cuda.current_stream(currents.device)
        rng_key = (bytes(generator.get_state().numpy()), steps, loss.n_res, currents.shape, currents.dtype, currents.device)
        if rng_key not in self._draws:
            start = perf_counter()
            if len(self._draws) >= 12:
                self._draws.popitem(last=False)
            self._draws[rng_key] = _ResidualSamples(generator, observations, steps, loss.n_res)
            stream.synchronize()
            self._counts["rng_setup_s"] += perf_counter() - start
        draws = self._draws[rng_key]
        leaves, spec = tree_flatten(params)
        key = (rng_key, stage, repr(spec), tuple((p.shape, p.stride(), p.dtype, p.device) for p in leaves),
               tuple(multipliers), tuple(rates), loss.use_vdata, loss.spatial_derivative_weight,
               loss.hard_bc, tuple(sorted(loss.w.items())), loss.n_res, loss.n_cells, loss.n_times,
               loss.n_fv_times, loss.residual, loss.obs.n_bins, loss.obs.n_gauss)
        if key not in self._graphs:
            start = perf_counter()
            if len(self._graphs) >= 5:
                self._graphs.popitem(last=False)
            self._graphs[key] = _AdamUpdate(params, function, draws, rates, multipliers, observations)
            stream.synchronize()
            self._counts["capture_s"] += perf_counter() - start
            self._counts["misses"] += 1
        else:
            self._counts["hits"] += 1
        update = self._graphs[key]
        update.reset(params, observations, draws)
        generator.set_state(draws.end_state)
        for _ in range(steps):
            logged = counter["step"] % log_every == 0
            if logged:
                record(update.current)
            update.graph.replay()
            counter["evals"] += 1
            if logged:
                attach(update.loss, update.parts)
            counter["step"] += 1
            if counter["step"] in snapshot_steps:
                snapshot(update.current)
        return update.independent_params(), False
