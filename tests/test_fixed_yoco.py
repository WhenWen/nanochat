from unittest.mock import patch

import torch

from nanochat.gpt import GPT, GPTConfig


def _kv_projection_inputs(fixed_yoco: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    config = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=4,
        n_head=1,
        n_kv_head=1,
        n_embd=32,
        window_pattern="L",
        fixed_yoco=fixed_yoco,
    )
    model = GPT(config)
    model.init_weights()

    # The default zero output projections make every layer see the same residual
    # direction at initialization. Nonzero projections expose the layer wiring.
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(generator=generator)
            block.mlp.c_proj.weight.normal_(generator=generator)

    key_inputs = []
    value_inputs = []
    hooks = []
    for block in model.transformer.h:
        hooks.append(block.attn.c_k.register_forward_pre_hook(lambda _module, args: key_inputs.append(args[0].detach().clone())))
        hooks.append(block.attn.c_v.register_forward_pre_hook(lambda _module, args: value_inputs.append(args[0].detach().clone())))

    with patch("nanochat.gpt.flash_attn.flash_attn_func", side_effect=lambda q, _k, _v, **_kwargs: q):
        model(torch.tensor([[1, 2, 3, 4]]))
    for hook in hooks:
        hook.remove()
    return key_inputs, value_inputs


def test_fixed_yoco_reuses_midpoint_for_upper_layer_keys():
    key_inputs, value_inputs = _kv_projection_inputs(fixed_yoco=True)

    assert not torch.equal(key_inputs[0], key_inputs[1])
    assert torch.equal(key_inputs[2], key_inputs[3])
    assert torch.equal(value_inputs[2], value_inputs[3])


def test_baseline_keeps_layer_local_key_inputs():
    key_inputs, value_inputs = _kv_projection_inputs(fixed_yoco=False)

    assert not torch.equal(key_inputs[2], key_inputs[3])
    assert not torch.equal(value_inputs[2], value_inputs[3])
