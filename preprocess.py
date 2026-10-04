"""Data loading, cleaning, encoding and normalisation for NSL-KDD style traffic records."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_srv_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]

ATTACK_GROUPS = {
    "DoS": "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb",
    "Probe": "satan ipsweep nmap portsweep mscan saint",
    "R2L": ("guess_passwd ftp_write imap phf multihop warezmaster warezclient spy xlock "
            "xsnoop snmpguess snmpgetattack httptunnel sendmail named"),
    "U2R": "buffer_overflow loadmodule rootkit perl sqlattack xterm ps",
}
ATTACK_MAP = {name: grp for grp, names in ATTACK_GROUPS.items() for name in names.split()}
_CLASS_LOOKUP = {c.lower(): c for c in CLASSES}


def map_label(value):
    """Map a raw NSL-KDD label (e.g. 'neptune') to one of CLASSES, or None if unknown."""
    s = str(value).strip().lower().rstrip(".")
    if s in _CLASS_LOOKUP:
        return _CLASS_LOOKUP[s]
    if s in ("anomaly", "attack"):
        return None
    return ATTACK_MAP.get(s)


def load_csv(path):
    """Read a CSV/TXT file with or without a header row. Extra columns are kept."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "protocol_type" not in df.columns:  # original NSL-KDD files have no header
        df = pd.read_csv(path, header=None, low_memory=False)
        if df.shape[1] < len(FEATURES):
            raise ValueError(
                f"Expected at least {len(FEATURES)} columns (NSL-KDD format), found {df.shape[1]}.")
        names = FEATURES + ["label", "difficulty"]
        extra = df.shape[1] - len(names)
        df.columns = names[: df.shape[1]] if extra <= 0 else names + [f"extra_{i}" for i in range(extra)]
    for alt in ("class", "attack", "Label"):
        if alt in df.columns and "label" not in df.columns:
            df = df.rename(columns={alt: "label"})
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else ""))
    return df


class Preprocessor:
    """Cleaning + one-hot encoding + min-max scaling, fitted once and reused for prediction."""

    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = MinMaxScaler()
        self.medians = None

    def _clean(self, df):
        num = df[NUMERIC].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        cat = df[CATEGORICAL].astype(str).apply(lambda s: s.str.strip().str.lower()).replace({"nan": "unknown", "": "unknown"})
        return num, cat

    def fit(self, df):
        num, cat = self._clean(df)
        self.medians = num.median().fillna(0)
        num = num.fillna(self.medians)
        self.encoder.fit(cat)
        self.scaler.fit(np.hstack([num.values, self.encoder.transform(cat)]))
        return self

    def transform(self, df):
        num, cat = self._clean(df)
        num = num.fillna(self.medians)
        x = self.scaler.transform(np.hstack([num.values, self.encoder.transform(cat)]))
        x = np.clip(x, 0.0, 1.0).astype("float32")
        return x.reshape((x.shape[0], x.shape[1], 1))  # (samples, features, 1) for Conv1D
