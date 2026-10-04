"""Create synthetic NSL-KDD-style data so the project runs without downloading anything.

For real results, download NSL-KDD (KDDTrain+.txt) and pass it to train_model.py --data.
"""
import os

import numpy as np
import pandas as pd

from preprocess import FEATURES, NUMERIC

BASE = os.path.dirname(os.path.abspath(__file__))
RATES = [c for c in NUMERIC if c.endswith("_rate")]


def _pick(rng, choices, p, n):
    return rng.choice(choices, size=n, p=p)


def _normal(rng, n):
    d = {f: np.zeros(n) for f in NUMERIC}
    d["duration"] = np.where(rng.random(n) < 0.7, 0, rng.exponential(30, n))
    d["src_bytes"] = rng.lognormal(5.5, 1.6, n)
    d["dst_bytes"] = rng.lognormal(7, 2, n)
    d["logged_in"] = (rng.random(n) < 0.75).astype(float)
    d["count"] = rng.integers(1, 30, n)
    d["srv_count"] = rng.integers(1, 30, n)
    d["hot"] = (rng.random(n) < 0.05) * rng.integers(1, 4, n)
    d["same_srv_rate"] = rng.beta(8, 1.5, n)
    d["diff_srv_rate"] = rng.beta(1, 15, n)
    d["srv_diff_srv_rate"] = rng.beta(1, 8, n)
    d["serror_rate"] = rng.beta(0.3, 12, n)
    d["srv_serror_rate"] = rng.beta(0.3, 12, n)
    d["rerror_rate"] = rng.beta(0.3, 12, n)
    d["srv_rerror_rate"] = rng.beta(0.3, 12, n)
    d["dst_host_count"] = rng.integers(60, 256, n)
    d["dst_host_srv_count"] = rng.integers(40, 256, n)
    d["dst_host_same_srv_rate"] = rng.beta(7, 2, n)
    d["dst_host_diff_srv_rate"] = rng.beta(1, 14, n)
    d["dst_host_same_src_port_rate"] = rng.beta(1.5, 6, n)
    d["dst_host_srv_diff_host_rate"] = rng.beta(1, 14, n)
    d["dst_host_serror_rate"] = rng.beta(0.3, 12, n)
    d["dst_host_srv_serror_rate"] = rng.beta(0.3, 12, n)
    d["dst_host_rerror_rate"] = rng.beta(0.3, 12, n)
    d["dst_host_srv_rerror_rate"] = rng.beta(0.3, 12, n)
    cat = {
        "protocol_type": _pick(rng, ["tcp", "udp", "icmp"], [0.80, 0.16, 0.04], n),
        "service": _pick(rng, ["http", "smtp", "ftp_data", "domain_u", "other", "ftp", "telnet", "ecr_i"],
                         [0.45, 0.14, 0.10, 0.12, 0.07, 0.05, 0.04, 0.03], n),
        "flag": _pick(rng, ["SF", "REJ", "S0", "RSTO"], [0.93, 0.03, 0.02, 0.02], n),
    }
    return d, cat, np.array(["normal"] * n, dtype=object)


def _attack(rng, n, group):
    d, cat, _ = _normal(rng, n)
    if group == "DoS":
        kind = _pick(rng, ["neptune", "smurf", "back"], [0.6, 0.3, 0.1], n)
        nep, smu, bak = kind == "neptune", kind == "smurf", kind == "back"
        d["src_bytes"] = np.where(nep, 0, np.where(smu, 1032, rng.normal(54540, 200, n)))
        d["dst_bytes"] = np.where(bak, rng.normal(8000, 500, n), 0)
        d["count"] = np.where(bak, rng.integers(1, 5, n), rng.integers(200, 512, n))
        d["srv_count"] = np.where(smu, rng.integers(450, 512, n), np.where(nep, rng.integers(5, 30, n), rng.integers(1, 5, n)))
        d["serror_rate"] = np.where(nep, rng.beta(30, 0.5, n), d["serror_rate"])
        d["srv_serror_rate"] = d["serror_rate"]
        d["dst_host_serror_rate"] = np.where(nep, rng.beta(30, 0.5, n), d["dst_host_serror_rate"])
        d["dst_host_srv_serror_rate"] = d["dst_host_serror_rate"]
        d["same_srv_rate"] = np.where(smu, 1.0, np.where(nep, rng.beta(1, 8, n), d["same_srv_rate"]))
        d["dst_host_srv_count"] = np.where(nep, rng.integers(1, 25, n), d["dst_host_srv_count"])
        d["logged_in"] = bak.astype(float)
        d["hot"] = bak * 2.0
        cat["protocol_type"] = np.where(smu, "icmp", "tcp")
        cat["service"] = np.where(smu, "ecr_i", np.where(bak, "http", "private"))
        cat["flag"] = np.where(nep, "S0", "SF")
        labels = kind.astype(object)
    elif group == "Probe":
        kind = _pick(rng, ["ipsweep", "portsweep", "satan", "nmap"], [0.3, 0.3, 0.25, 0.15], n)
        ips = kind == "ipsweep"
        d["src_bytes"] = np.where(ips, 8, rng.integers(0, 20, n))
        d["dst_bytes"] = np.zeros(n)
        d["duration"] = np.zeros(n)
        d["count"] = rng.integers(1, 20, n)
        d["srv_count"] = rng.integers(1, 10, n)
        d["diff_srv_rate"] = rng.beta(6, 2, n)
        d["srv_diff_srv_rate"] = rng.beta(4, 2, n)
        d["same_srv_rate"] = rng.beta(1.5, 6, n)
        d["rerror_rate"] = rng.beta(5, 2, n)
        d["srv_rerror_rate"] = d["rerror_rate"]
        d["dst_host_count"] = rng.integers(1, 256, n)
        d["dst_host_srv_count"] = rng.integers(1, 20, n)
        d["dst_host_diff_srv_rate"] = rng.beta(5, 2, n)
        d["dst_host_srv_diff_host_rate"] = rng.beta(4, 2, n)
        d["dst_host_rerror_rate"] = rng.beta(4, 2, n)
        d["dst_host_srv_rerror_rate"] = d["dst_host_rerror_rate"]
        d["logged_in"] = np.zeros(n)
        cat["protocol_type"] = np.where(ips, "icmp", "tcp")
        cat["service"] = np.where(ips, "eco_i", _pick(rng, ["private", "http", "telnet", "other"], [0.5, 0.2, 0.15, 0.15], n))
        cat["flag"] = np.where(ips, "SF", _pick(rng, ["REJ", "RSTR", "SH", "S0"], [0.5, 0.2, 0.15, 0.15], n))
        labels = kind.astype(object)
    elif group == "R2L":
        kind = _pick(rng, ["guess_passwd", "warezclient", "ftp_write"], [0.5, 0.4, 0.1], n)
        gp, wc = kind == "guess_passwd", kind == "warezclient"
        d["duration"] = rng.exponential(60, n)
        d["src_bytes"] = np.where(wc, rng.lognormal(7, 1, n), rng.integers(100, 130, n))
        d["dst_bytes"] = np.where(wc, rng.lognormal(8, 1, n), rng.integers(100, 400, n))
        d["num_failed_logins"] = gp * rng.integers(1, 6, n)
        d["hot"] = wc * rng.integers(10, 30, n) + (kind == "ftp_write") * rng.integers(1, 5, n)
        d["is_guest_login"] = wc.astype(float)
        d["logged_in"] = (~gp).astype(float)
        d["count"] = rng.integers(1, 6, n)
        d["srv_count"] = rng.integers(1, 6, n)
        d["dst_host_count"] = rng.integers(1, 40, n)
        cat["protocol_type"] = np.array(["tcp"] * n)
        cat["service"] = np.where(gp, "telnet", np.where(wc, "ftp_data", "ftp"))
        cat["flag"] = np.array(["SF"] * n)
        labels = kind.astype(object)
    else:  # U2R
        kind = _pick(rng, ["buffer_overflow", "rootkit", "loadmodule"], [0.55, 0.25, 0.2], n)
        d["duration"] = rng.exponential(120, n)
        d["src_bytes"] = rng.integers(500, 3000, n)
        d["dst_bytes"] = rng.integers(1500, 6000, n)
        d["logged_in"] = np.ones(n)
        d["root_shell"] = (rng.random(n) < 0.8).astype(float)
        d["su_attempted"] = (rng.random(n) < 0.2).astype(float)
        d["num_root"] = rng.integers(0, 8, n)
        d["num_shells"] = rng.integers(1, 3, n)
        d["num_file_creations"] = rng.integers(0, 6, n)
        d["num_compromised"] = rng.integers(0, 4, n)
        d["hot"] = rng.integers(1, 6, n)
        d["count"] = np.ones(n)
        d["srv_count"] = np.ones(n)
        cat["protocol_type"] = np.array(["tcp"] * n)
        cat["service"] = np.array(["telnet"] * n)
        cat["flag"] = np.array(["SF"] * n)
        labels = kind.astype(object)
    return d, cat, labels


def make_dataset(counts, seed=7, label=True):
    rng = np.random.default_rng(seed)
    frames = []
    for group, n in counts.items():
        if n == 0:
            continue
        d, cat, lab = _normal(rng, n) if group == "normal" else _attack(rng, n, group)
        if group != "normal":  # ~6% of attack rows look like ordinary traffic, to keep it realistic
            hard = rng.random(n) < 0.06
            if hard.any():
                nd, nc, _ = _normal(rng, n)
                for k in d:
                    d[k] = np.where(hard, nd[k], d[k])
                for k in cat:
                    cat[k] = np.where(hard, nc[k], cat[k])
        df = pd.DataFrame({**d, **cat})
        df["label"] = lab
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    for c in RATES:
        df[c] = df[c].clip(0, 1).round(2)
    for c in NUMERIC:
        if c not in RATES:
            df[c] = df[c].clip(lower=0).round(0).astype(int)
    df = df[FEATURES + ["label"]]
    return df if label else df.drop(columns="label")


def main():
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    make_dataset({"normal": 5000, "DoS": 4000, "Probe": 1800, "R2L": 900, "U2R": 300}, seed=7) \
        .to_csv(os.path.join(BASE, "data", "sample_train.csv"), index=False)
    mix = {"normal": 180, "DoS": 70, "Probe": 30, "R2L": 14, "U2R": 6}
    make_dataset(mix, seed=21).to_csv(os.path.join(BASE, "data", "sample_upload_labeled.csv"), index=False)
    make_dataset(mix, seed=33, label=False).to_csv(os.path.join(BASE, "data", "sample_upload_unlabeled.csv"), index=False)
    print("Wrote data/sample_train.csv, sample_upload_labeled.csv, sample_upload_unlabeled.csv")


if __name__ == "__main__":
    main()
