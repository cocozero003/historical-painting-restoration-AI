
#!/usr/bin/env python3
import argparse, pathlib, urllib.request
URL="https://huggingface.co/saic-mdal/lama/resolve/main/big-lama.pt"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dest", default="models/lama")
    a=ap.parse_args(); d=pathlib.Path(a.dest); d.mkdir(parents=True, exist_ok=True)
    out=d/"big-lama.pt"; print("Downloading LaMa to", out); urllib.request.urlretrieve(URL, out); print("Done.")
if __name__=="__main__": main()
