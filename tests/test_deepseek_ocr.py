"""DeepSeek OCR backend: its layout output becomes the same tokens RapidOCR gives, and any API failure falls
back to the local reader."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import dimsolve as D  # noqa: E402

PAGE = ("<｜ref｜>BEDROOM<｜/ref｜><｜box｜>[[100, 200, 200, 220]]<｜/box｜>\n"
        "<｜ref｜>3600<｜/ref｜><｜box｜>[[500, 50, 560, 70]]<｜/box｜>\n"
        "<｜ref｜>KITCHEN\n12'-0\" x 10'-0\"<｜/ref｜><｜box｜>[[300, 600, 400, 660]]<｜/box｜>")


def test_layout_text_becomes_pixel_boxes():
    res = D.parse_deepseek_ocr(PAGE, 2000, 1000)
    texts = [t for _, t, _ in res]
    assert texts == ["BEDROOM", "3600", "KITCHEN", "12'-0\" x 10'-0\""]
    q = res[1][0]
    assert q[0] == [1000.0, 50.0] and q[2] == [1120.0, 70.0]          # 0-1000 -> pixels of this image
    assert res[2][0][2][1] == res[3][0][0][1] == 630.0                 # two-line block split into two rows


class FakeAPI(D.DeepSeekOCR):
    def __init__(self, reply=None, fail=False):
        super().__init__("k", lambda: (lambda img: ([([[0, 0], [1, 0], [1, 1], [0, 1]], "LOCAL", 0.5)], None)))
        self.reply, self.fail = reply, fail

    def _post(self, png):
        if self.fail:
            raise OSError("network down")
        return {"pages": [{"index": 0, "text": self.reply}]}


def test_tokens_from_the_api():
    img = np.full((1000, 2000, 3), 255, np.uint8)
    toks = D.ocr_tokens(img, FakeAPI(PAGE))
    nums = [t for t in toks if t["kind"] == "num"]
    assert [t["value"] for t in nums] == [3600.0] and abs(nums[0]["x"] - 1060) < 1
    assert any(t["kind"] == "size" for t in toks)


def test_api_failure_reads_locally():
    api = FakeAPI(fail=True)
    res = D._ocr_tiled(api, np.full((500, 500, 3), 255, np.uint8))
    assert [t for _, t, _ in res] == ["LOCAL"] and api.failures == 1


def test_default_is_local_without_key(monkeypatch):
    monkeypatch.setenv("PLANFORGE_OCR", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(D, "_rapid", lambda: "rapid")
    assert D.default_ocr() == "rapid"
