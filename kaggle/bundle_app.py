"""Packs app/ into the APP_ZIP block of kaggle/run_planforge_kaggle.py, so the Kaggle runner always carries
the current engine and website.

    python kaggle/bundle_app.py           # rebuild the bundle (run after changing anything in app/)
    python kaggle/bundle_app.py --check   # exit 1 if the bundle is out of date (used by CI)
"""
import base64
import io
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
RUNNER = os.path.join(ROOT, "kaggle", "run_planforge_kaggle.py")
BLOCK = re.compile(r'APP_ZIP = """\n(.*?)"""', re.S)
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_EXT = {".pyc", ".pt", ".pth", ".onnx", ".safetensors", ".npy"}


def app_files():
    out = {}
    for d, dirs, files in os.walk(APP):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files):
            if os.path.splitext(f)[1] in SKIP_EXT:
                continue
            p = os.path.join(d, f)
            out[os.path.relpath(p, APP).replace(os.sep, "/")] = open(p, "rb").read()
    return out


def bundled_files(text):
    m = BLOCK.search(text)
    if not m:
        sys.exit("APP_ZIP block not found in " + RUNNER)
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(m.group(1).split()))))
    return {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}


def build_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))   # fixed time: same input, same bundle
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, files[name])
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "\n".join(b64[i:i + 100] for i in range(0, len(b64), 100)) + "\n"


def main():
    text = open(RUNNER).read()
    want = app_files()
    have = bundled_files(text)
    if "--check" in sys.argv:
        stale = sorted(n for n in set(want) | set(have) if want.get(n) != have.get(n))
        if stale:
            print("Kaggle bundle is out of date ({} file(s) differ): {}".format(len(stale), ", ".join(stale[:10])))
            print("Fix: python kaggle/bundle_app.py")
            sys.exit(1)
        print("Kaggle bundle is up to date ({} files)".format(len(want)))
        return
    new = BLOCK.sub(lambda m: 'APP_ZIP = """\n' + build_zip(want) + '"""', text, count=1)
    open(RUNNER, "w").write(new)
    print("bundled {} files from app/ into {}".format(len(want), os.path.relpath(RUNNER, ROOT)))


if __name__ == "__main__":
    main()
