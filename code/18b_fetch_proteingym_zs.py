#!/usr/bin/env python3
"""
18b_fetch_proteingym_zs.py — Rebuild data/proteingym_zs/ from the Zenodo archive.

The ProteinGym v1.3 zero-shot score archive is 1.78 GiB, but the published-
predictor comparison of Supplementary Table S11 needs only five of its members.
This script reads the remote zip's central directory over HTTP Range and
decompresses just those five entries, writing them byte-identically to
data/proteingym_zs/. The shipped copies were obtained this way on 2026-11-13
(38.1 MB fetched, 23 requests, 108 s).

Usage:
  python 18b_fetch_proteingym_zs.py --list      # show the matched entries
  python 18b_fetch_proteingym_zs.py --fetch     # download and overwrite the 5 CSVs
  python 18b_fetch_proteingym_zs.py --verify    # check md5 against checksums.md5

Network note: where `zenodo.org` does not resolve (poisoned DNS), connect to an
IP from the record's own DNS answer while keeping the real hostname for SNI and
certificate validation; pass it with `--ip`.
"""
import argparse
import hashlib
import http.client
import io
import os
import socket
import ssl
import sys
import zipfile

RECORD = "15293562"
FILE = "zero_shot_substitutions_scores.zip"
HOST = "zenodo.org"
DEFAULT_IP = "188.185.42.113"
URL = "/api/records/%s/files/%s/content" % (RECORD, FILE)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.environ.get("PG_ZS_DIR", os.path.join(BASE, "data",
                                                  "proteingym_zs"))

ASSAYS = {
    "TEM-1": ("BLAT_ECOLX_Stiffler_2015", "BLAT_ECOLX_Stiffler_2015.csv"),
    "VIM-2": ("A4GRB6_PSEAI_Chen_2020", "A4GRB6_PSEAI_Chen_2020.csv"),
    "AAC(3)-I": ("AACC1_PSEAI_Dandage_2018", "AACC1_PSEAI_Dandage_2018.csv"),
    "APH(3')-IIa": ("KKA2_KLEPN_Melnikov_2014", "KKA2_KLEPN_Melnikov_2014.csv"),
    "DHFR": ("DYR_ECOLI_Nguyen_2023", "DYR_ECOLI_Nguyen_2023.csv"),
}


class _Conn(http.client.HTTPSConnection):
    """HTTPS to a fixed IP, but with the real hostname for SNI and certs."""

    def __init__(self, host, ip, timeout=90):
        super().__init__(host, 443, timeout=timeout)
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, 443), self.timeout)
        ctx = ssl.create_default_context()
        self.sock = ctx.wrap_socket(sock, server_hostname=self.host)


class RemoteZip(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range, 1 MiB block cache."""

    BLOCK = 1 << 20

    def __init__(self, url, ip):
        super().__init__()
        self.url, self.ip = url, ip
        self.pos = 0
        self.cache = {}
        self.size = int(self._head_size())
        self.nreq = 0
        self.nbyte = 0

    # -- http ------------------------------------------------------------
    def _get(self, rng):
        c = _Conn(HOST, self.ip)
        c.request("GET", self.url, headers={"Range": rng,
                                            "User-Agent": "amr-audit/1.0"})
        r = c.getresponse()
        if r.status not in (200, 206):
            raise RuntimeError("HTTP %s for %s" % (r.status, rng))
        data = r.read()
        c.close()
        self.nreq += 1
        self.nbyte += len(data)
        return data

    def _head_size(self):
        c = _Conn(HOST, self.ip)
        c.request("HEAD", self.url, headers={"User-Agent": "amr-audit/1.0"})
        r = c.getresponse()
        n = r.getheader("Content-Length")
        c.close()
        return n

    def _block(self, idx):
        if idx not in self.cache:
            lo = idx * self.BLOCK
            hi = min(lo + self.BLOCK - 1, self.size - 1)
            self.cache[idx] = self._get("bytes=%d-%d" % (lo, hi))
            if len(self.cache) > 24:                       # keep memory bounded
                for k in list(self.cache)[:-16]:
                    self.cache.pop(k, None)
        return self.cache[idx]

    # -- file protocol ---------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = []
        while n > 0:
            idx, off = divmod(self.pos, self.BLOCK)
            blk = self._block(idx)
            take = min(n, len(blk) - off)
            if take <= 0:
                break
            out.append(blk[off:off + take])
            self.pos += take
            n -= take
        return b"".join(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify():
    """Compare data/proteingym_zs/*.csv against checksums.md5."""
    man = os.path.join(OUTDIR, "checksums.md5")
    if not os.path.exists(man):
        print("no checksums.md5 in %s" % OUTDIR)
        return 1
    bad = 0
    with open(man) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            want, name = line.split()[0], line.split()[-1]
            path = os.path.join(OUTDIR, name)
            if not os.path.exists(path):
                print("  %-34s MISSING" % name)
                bad += 1
                continue
            got = _md5(path)
            ok = got == want
            bad += 0 if ok else 1
            print("  %-34s %s" % (name, "OK  " + got if ok
                                  else "FAIL %s != %s" % (got, want)))
    print("verify: %s" % ("ALL OK" if bad == 0 else "%d bad" % bad))
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        return verify()

    rf = RemoteZip(URL, a.ip)
    print("remote size: %.2f GiB" % (rf.size / 2**30))
    zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
    names = zf.namelist()
    print("entries:", len(names))

    # Layout: one CSV per DMS assay, with all model score columns side by side.
    want = {}
    for lab, (did, _fn) in ASSAYS.items():
        pats = [n for n in names
                if n == did + ".csv" or n.endswith("/" + did + ".csv")]
        want[lab] = pats[0] if pats else None
    miss = sorted(k for k, v in want.items() if v is None)
    print("matched %d / %d assays; missing %d"
          % (len(want) - len(miss), len(want), len(miss)))
    for k in miss:
        print("   MISS", k)

    if a.list:
        tot = sum(zf.getinfo(v).file_size for v in want.values() if v)
        print("uncompressed bytes needed: %.1f MB" % (tot / 1e6))
        return 0

    if not a.fetch:
        return 0

    os.makedirs(OUTDIR, exist_ok=True)
    for lab, (did, fn) in ASSAYS.items():
        src = want[lab]
        data = zf.read(src)
        with open(os.path.join(OUTDIR, fn), "wb") as fh:
            fh.write(data)
        print("wrote %-32s %.2f MB  md5 %s" % (fn, len(data) / 1e6,
                                              hashlib.md5(data).hexdigest()))
    print("requests=%d bytes=%.1f MB" % (rf.nreq, rf.nbyte / 1e6))
    return verify()


if __name__ == "__main__":
    sys.exit(main())
