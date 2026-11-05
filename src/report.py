"""Betiklerin ortak rapor yazıcısı.

T-01'de scripts/01_veri_kesfi.py içindeydi; T-02 betiği de kullandığı için buraya taşındı.
"""

from __future__ import annotations

from pathlib import Path


class Report:
    """Ekrana yazar ve aynı içeriği markdown dosyası için biriktirir."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def h(self, title: str) -> None:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        self.lines.append(f"\n## {title}\n")

    def p(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def table(self, obj) -> None:
        text = obj.to_string() if hasattr(obj, "to_string") else str(obj)
        print(text)
        self.lines.append("```\n" + text + "\n```")

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def short(text, n: int = 90) -> str:
    """Uzun metni tek satıra indirip kısaltır."""
    text = str(text).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def pct(part: float, whole: float) -> str:
    return f"%{100 * part / whole:.2f}" if whole else "-"


def use_utf8_stdout() -> None:
    """Windows konsolunda emoji yüzünden çökmemek için."""
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
