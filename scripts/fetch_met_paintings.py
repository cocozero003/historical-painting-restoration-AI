
#!/usr/bin/env python3
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
SEARCH_URL="https://collectionapi.metmuseum.org/public/collection/v1/search"
OBJ_URL="https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
def fetch_ids(q="painting"):
    r=requests.get(SEARCH_URL, params={"q":q,"hasImages":"true"}, timeout=30); r.raise_for_status(); d=r.json()
    return d.get("objectIDs",[]) or []
def in_range(o, lo, hi):
    try: b=int(o.get("objectBeginDate") or 0); e=int(o.get("objectEndDate") or 0)
    except Exception: return False
    return not (e<lo or b>hi)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--min_year", type=int, default=1300)
    ap.add_argument("--max_year", type=int, default=1650); ap.add_argument("--limit", type=int, default=120)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ids=fetch_ids(); print("Candidates:", len(ids))
    n=0
    for oid in tqdm(ids, total=len(ids)):
        if n>=a.limit: break
        try:
            r=requests.get(OBJ_URL.format(oid=oid), timeout=30)
            if r.status_code!=200: continue
            o=r.json()
            if not o.get("isPublicDomain",False): continue
            if not in_range(o, a.min_year, a.max_year): continue
            url=o.get("primaryImage") or o.get("primaryImageSmall")
            if not url: continue
            name=(o.get("title") or f"met_{oid}").strip().replace(" ","_")[:80]+"_"+str(oid)+".jpg"
            img=requests.get(url, timeout=60)
            if img.status_code==200 and img.content:
                (out/name).write_bytes(img.content); n+=1
        except Exception: continue
    print("Saved", n, "images to", out)
if __name__=="__main__": main()
