
#!/usr/bin/env python3
import argparse, pathlib, urllib.request
URL="https://huggingface.co/datasets/hf-internal-testing/dummy-pytorch-files/resolve/main/state_dict.pt"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dest", default="models/unet")
    a=ap.parse_args(); d=pathlib.Path(a.dest); d.mkdir(parents=True, exist_ok=True)
    out=d/"damage_mask_unet.pth"; print("Downloading demo U-Net weights to", out); urllib.request.urlretrieve(URL, out); print("Done.")
if __name__=="__main__": main()
