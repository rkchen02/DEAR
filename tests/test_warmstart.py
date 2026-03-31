import torch
import torch.nn as nn

from gnarl_transfer.warmstart import copy_matching_state_dict


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)


def test_copy_matching_state_dict_copies_identical_shapes():
    source = TinyNet()
    target = TinyNet()

    with torch.no_grad():
        for _, param in source.named_parameters():
            param.fill_(1.5)
        for _, param in target.named_parameters():
            param.zero_()

    result = copy_matching_state_dict(target, source.state_dict())

    assert "fc1.weight" in result["copied"]
    assert "fc1.bias" in result["copied"]
    assert "fc2.weight" in result["copied"]
    assert "fc2.bias" in result["copied"]

    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        assert torch.allclose(src_param, tgt_param)