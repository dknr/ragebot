#!/usr/bin/env python3
"""Download the ONNX Runtime NuGet package and extract the arch-matched shared lib."""
import argparse
import os
import platform
import sys
import urllib.request
import zipfile

VERSION = "1.30.0"
URL = f"https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/{VERSION}"

# nuget runtime dir key -> uname -m
ARCHMAP = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    for attempt in range(5):
        print(f"downloading {url} (attempt {attempt + 1})")
        with urllib.request.urlopen(url) as r:
            want = int(r.headers.get("Content-Length") or 0)
            got = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
        if want and got == want:
            os.replace(tmp, path)
            return
        print(f"  truncated: got {got}, expected {want}; retrying")
    raise RuntimeError(f"failed to download {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="dir for the extracted .so")
    ap.add_argument("--cache", default="work", help="dir for the .nupkg cache")
    args = ap.parse_args()

    arch = ARCHMAP.get(platform.machine(), platform.machine())
    nupkg = os.path.join(args.cache, "ort.nupkg")
    download(URL, nupkg)

    member = f"runtimes/linux-{arch}/native/libonnxruntime.so"
    with zipfile.ZipFile(nupkg) as z:
        if member not in z.namelist():
            sys.exit(f"error: {member} not in package; available: {[n for n in z.namelist() if 'native' in n]}")
        data = z.read(member)
    out = os.path.join(args.out, "ort.so")
    with open(out, "wb") as f:
        f.write(data)
    print(f"{out}: {len(data)} bytes (linux-{arch})")


if __name__ == "__main__":
    main()
