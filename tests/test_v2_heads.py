"""v2 training heads: wall type labels derived from room maps, CubiCasa floor-level labels, growing the v1
(27-channel) head to 32 channels, and losses that skip unlabelled pixels. Needs PyTorch (skipped in CI)."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
import fp_common as F  # noqa: E402

W, BG, OUT, BED = (F.ROOM_CLASSES.index(n) for n in ("Wall", "Background", "Outdoor", "Bedroom"))


def two_room_house():
    room = np.full((120, 160), BG, np.uint8)
    room[20:100, 20:140] = W                      # building with 6 px walls
    room[26:94, 26:77] = BED
    room[26:94, 83:134] = BED                     # partition at x 77..83
    return room


def test_exterior_and_interior_walls():
    lab = F.wall_type_labels(two_room_house())
    assert lab[60, 22] == 1 and lab[22, 50] == 1          # outer walls
    assert lab[60, 80] == 0                               # the partition in the middle
    assert lab[60, 50] == F.IGNORE and lab[5, 5] == F.IGNORE   # floor and outside: not walls


def test_wall_next_to_balcony_is_exterior():
    room = two_room_house()
    room[26:94, 83:134] = OUT                     # right room is a balcony
    lab = F.wall_type_labels(room)
    assert lab[60, 80] == 1


def test_no_outside_means_unknown():
    room = np.full((60, 60), BED, np.uint8)
    room[:, 28:32] = W
    assert (F.wall_type_labels(room) == F.IGNORE).all()


def test_head_grows_from_v1_weights():
    m1 = __import__("segmentation_models_pytorch").Unet("resnet18", encoder_weights=None, classes=F.N_BASE)
    m2 = __import__("segmentation_models_pytorch").Unet("resnet18", encoder_weights=None, classes=F.N_OUT)
    st, grown = F.expand_head(m1.state_dict(), m2)
    assert grown == 27
    m2.load_state_dict(st)
    w1, w2 = m1.segmentation_head[0].weight, m2.segmentation_head[0].weight
    assert torch.equal(w2[:27], w1)


def test_losses_skip_unlabelled():
    out = torch.randn(2, F.N_OUT, 32, 32, requires_grad=True)
    room = torch.randint(0, F.N_ROOM, (2, 32, 32))
    icon = torch.randint(0, F.N_ICON, (2, 32, 32))
    heat = torch.full((2, 3, 32, 32), -1.0)
    ign = torch.full((2, 32, 32), F.IGNORE)
    crit = F.MultiTaskLoss()
    total, parts = crit(out, room, icon, heat, ign, ign)
    assert parts["wtype"].item() == 0 and parts["level"].item() == 0
    lv = ign.clone()
    lv[:, :8] = 1
    total2, parts2 = crit(out, room, icon, heat, ign, lv)
    assert parts2["level"].item() > 0
    total2.backward()
    assert out.grad[:, F.N_BASE + F.N_WT:].abs().sum() > 0      # level channels learn
    assert out.grad[:, F.N_BASE:F.N_BASE + F.N_WT].abs().sum() == 0   # wall type: nothing labelled


def test_v1_loss_call_still_works():
    out = torch.randn(1, F.N_BASE, 32, 32)
    total, parts = F.MultiTaskLoss()(out, torch.zeros(1, 32, 32, dtype=torch.long),
                                     torch.zeros(1, 32, 32, dtype=torch.long), torch.full((1, 3, 32, 32), -1.0))
    assert "wtype" not in parts


def test_cubicasa_level_labels(tmp_path):
    pytest.importorskip("svgpathtools")
    import fp_prep
    sys.path.insert(0, str(tmp_path))
    try:
        fp_prep.setup_parser(str(tmp_path / "parser"))
        sys.path.insert(0, str(tmp_path / "parser"))
    except Exception:
        pytest.skip("CubiCasa parser not reachable")
    svg = ('<svg xmlns="http://www.w3.org/2000/svg"><g class="Space LivingRoom"><polygon points="10,10 90,10 90,90 '
           '10,90 "/></g><g class="Space Elevated"><polygon points="20,20 40,20 40,40 20,40 "/></g>'
           '<g class="Space OpenToBelow"><polygon points="60,60 80,60 80,80 60,80 "/></g></svg>')
    p = tmp_path / "m.svg"
    p.write_text(svg)
    lv, n = fp_prep._parse_levels(str(p), (100, 100))
    assert n == 2 and lv[30, 30] == 1 and lv[70, 70] == 2 and lv[50, 15] == 0 and lv[5, 5] == 255
