"""Ham veriyi indirme ve okuma.

T-01'de bu kod scripts/01_veri_kesfi.py içindeydi. T-02'de ikinci bir betik de
aynı veriyi okuyacağı için buraya taşındı: tek yerde dursun, iki betik aynı
şekilde okusun.
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "raw" / "TRSAv1.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

DATA_URL = (
    "https://huggingface.co/datasets/maydogan/"
    "Turkish_SentimentAnalysis_TRSAv1/resolve/main/TRSAv1.csv"
)
DATA_PAGE = "https://huggingface.co/datasets/maydogan/Turkish_SentimentAnalysis_TRSAv1/tree/main"

# T-01'de doğrulandı: TRSAv1.csv kolonları id, score, review
ID_COL, LABEL_COL, TEXT_COL = "id", "score", "review"


def ensure_data(path: Path = DEFAULT_CSV, allow_download: bool = True) -> None:
    """Dosya yoksa Hugging Face'ten indirir."""
    if path.exists():
        return
    if not allow_download:
        sys.exit(f"Veri bulunamadı: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"Veri indiriliyor (~16 MB):\n  {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, tmp)
    except Exception as exc:  # ağ hatası, proxy vb.
        tmp.unlink(missing_ok=True)
        sys.exit(
            f"\nİndirme başarısız: {exc}\n"
            f"Elle indir: {DATA_PAGE}\n"
            f"Dosyayı şuraya koy: {path}"
        )
    tmp.replace(path)
    print("İndirme tamam.\n")


def read_csv_robust(path: Path) -> tuple[pd.DataFrame, str, str]:
    """Kodlamayı ve ayırıcıyı deneyerek CSV'yi okur. (df, kodlama, ayırıcı) döner."""
    raw = path.read_bytes()
    head = raw[:200].lstrip().lower()
    if head.startswith((b"<!doctype", b"<html", b"version https://git-lfs")):
        sys.exit(f"{path.name} gerçek bir CSV değil (HTML ya da LFS işaretçisi). Silip elle indir: {DATA_PAGE}")

    text, encoding = None, None
    for enc in ("utf-8-sig", "cp1254"):  # utf-8-sig, BOM'lu ve BOM'suz UTF-8'i okur
        try:
            text, encoding = raw.decode(enc), enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text, encoding = raw.decode("utf-8", errors="replace"), "utf-8 (bozuk baytlar değiştirildi)"

    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
        except pd.errors.ParserError:
            continue
        if df.shape[1] >= 2:
            return df, encoding, sep
    sys.exit("CSV okunamadı: virgül, noktalı virgül ve sekme ayırıcılarının hiçbiri işe yaramadı.")


def load_processed(name: str) -> pd.DataFrame:
    """02 betiğinin ürettiği parçalardan birini okur: train, val, test veya zor_ornekler.

    T-03'te scripts/03_baseline.py içindeydi; T-04 de kullandığı için buraya taşındı.
    keep_default_na=False: metni "NA", "null", "None" olan yorumlar boş değer sanılmasın.
    """
    path = PROCESSED / f"{name}.csv"
    if not path.exists():
        sys.exit(f"{path} yok. Önce çalıştır: python scripts/02_temizlik_bolme.py")
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def load_trsa(path: Path = DEFAULT_CSV, allow_download: bool = True) -> pd.DataFrame:
    """TRSAv1'i standart kolon adlarıyla döndürür: id, text, label.

    Etiketlerdeki baştaki/sondaki boşluklar temizlenir; boş metin veya etiketi
    olan satırlar atılır (T-01'de TRSAv1'de hiç olmadığı görüldü).
    """
    ensure_data(path, allow_download)
    df, _, _ = read_csv_robust(path)
    missing = {ID_COL, LABEL_COL, TEXT_COL} - set(df.columns)
    if missing:
        sys.exit(f"Beklenen kolonlar yok: {missing}. Mevcut kolonlar: {list(df.columns)}")
    df = df[[ID_COL, TEXT_COL, LABEL_COL]].dropna()
    df.columns = ["id", "text", "label"]
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    return df.reset_index(drop=True)
