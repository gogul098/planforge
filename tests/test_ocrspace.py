"""OCR.space backend: its word boxes become the same texts RapidOCR gives (rows split at wide gaps, vertical
dimensions read from a turned copy), images are kept under the free 1 MB limit, and any failure reads locally."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
import dimsolve as D  # noqa: E402


def W(t, l, top, w, h=20):
    return {"WordText": t, "Left": l, "Top": top, "Width": w, "Height": h}


REPLY = {"OCRExitCode": 1, "IsErroredOnProcessing": False, "ParsedResults": [{"TextOverlay": {"Lines": [
    {"LineText": "BED ROOM", "Words": [W("BED", 100, 100, 40), W("ROOM", 150, 100, 60)]},
    {"LineText": "3600 4200", "Words": [W("3600", 300, 400, 50), W("4200", 900, 400, 50)]},
    {"LineText": "12'-0\" x 10'-0\"", "Words": [W("12'-0\"", 100, 200, 60), W("x", 165, 200, 10),
                                                W("10'-0\"", 180, 200, 60)]}]}}]}


def test_words_become_texts_split_at_wide_gaps():
    res = D.parse_ocrspace(REPLY)
    assert [t for _, t, _ in res] == ["BED ROOM", "3600", "4200", "12'-0\" x 10'-0\""]
    assert res[2][0][0] == [900.0, 400.0] and res[2][0][2] == [950.0, 420.0]


class Fake(D.OCRSpaceOCR):
    def __init__(self, replies=None, fail=None, rotated=False):
        super().__init__("k", lambda: (lambda img: ([([[0, 0], [1, 0], [1, 1], [0, 1]], "LOCAL", 0.5)], None)),
                         rotated=rotated)
        self.replies, self.fail, self.sent = list(replies or []), fail, []

    def _post(self, payload, name):
        self.sent.append(len(payload))
        if self.fail:
            raise RuntimeError(self.fail)
        return self.replies.pop(0)


def test_tokens_from_the_api():
    toks = D.ocr_tokens(np.full((600, 1200, 3), 255, np.uint8), Fake([REPLY]))
    assert sorted(t["value"] for t in toks if t["kind"] == "num") == [3600.0, 4200.0]
    assert any(t["kind"] == "size" for t in toks)


def test_vertical_dimension_from_the_turned_copy():
    H, Wd = 1000, 800
    upright = {"OCRExitCode": 1, "ParsedResults": [{"TextOverlay": {"Lines": [
        {"Words": [W("KITCHEN", 300, 300, 100)]}]}}]}
    # in the turned image (Wd x H), '2400' at u 400..460, v 50..70 -> page x = Wd-1-v (729..749), y = u (400..460)
    turned = {"OCRExitCode": 1, "ParsedResults": [{"TextOverlay": {"Lines": [
        {"Words": [W("2400", 400, 50, 60)]},                 # vertical on the page: kept
        {"Words": [W("KITCHEN", 300, 400, 100)]}]}}]}        # horizontal on the page (a re-read): dropped
    res, _ = Fake([upright, turned], rotated=True)(np.full((H, Wd, 3), 255, np.uint8))
    assert [t for _, t, _ in res] == ["KITCHEN", "2400"]
    q = res[1][0]
    assert q[0] == [729.0, 400.0] and q[2] == [749.0, 460.0]
    tok = [t for t in D.ocr_tokens(np.full((H, Wd, 3), 255, np.uint8), Fake([upright, turned], rotated=True))
           if t["kind"] == "num"][0]
    assert tok["value"] == 2400 and abs(tok["x"] - 739) < 1 and abs(tok["y"] - 430) < 1


def test_daily_limit_or_any_error_reads_locally():
    api = Fake(fail="You may only perform this action upto maximum 500 number of times within 86400 seconds")
    res = D._ocr_tiled(api, np.full((400, 400, 3), 255, np.uint8))
    assert [t for _, t, _ in res] == ["LOCAL"] and api.failures == 1


def test_big_noisy_image_kept_under_one_megabyte():
    rng = np.random.default_rng(0)
    img = (rng.random((2500, 2500, 3)) * 255).astype(np.uint8)
    data, name, f = D._encode_under(img, D.OCRSPACE_MAX_BYTES)
    assert len(data) <= D.OCRSPACE_MAX_BYTES and name.endswith(".jpg")


def test_choice_by_setting(monkeypatch):
    monkeypatch.setattr(D, "_rapid", lambda: "rapid")
    monkeypatch.setattr(D, "_REMOTE", [])
    monkeypatch.setenv("PLANFORGE_OCR", "ocrspace")
    monkeypatch.delenv("OCRSPACE_API_KEY", raising=False)
    assert D.default_ocr() == "rapid"                       # no key: local
    monkeypatch.setenv("OCRSPACE_API_KEY", "abc")
    assert isinstance(D.default_ocr(), D.OCRSpaceOCR)
