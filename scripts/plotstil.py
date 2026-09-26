"""Gemeinsamer Stil für alle matplotlib-Grafiken.

Die Farben kommen aus der Brand-Datei, die in _quarto.yml ausgewählt ist
(brand/htl.yml, brand/hak.yml, brand/white.yml oder brand/dark.yml).
Wechselt man das Design, passen sich die Grafiken automatisch an.

Beim dunklen Design bekommen die Grafiken auf den Folien helle Schrift und
einen transparenten Hintergrund, im (immer weißen) Handout dunkle Schrift.

Verwendung in einem Kapitel:

    import matplotlib.pyplot as plt
    from plotstil import PRIMAER, SEKUNDAER, TEXT, GRAU
"""

import json
import os
from pathlib import Path

import matplotlib as mpl
import yaml

ROOT = Path(os.environ.get("QUARTO_PROJECT_DIR", Path(__file__).resolve().parent.parent))


def _format() -> str:
    """'revealjs' oder 'typst' – je nachdem, was Quarto gerade rendert."""
    info = os.environ.get("QUARTO_EXECUTE_INFO")
    if info and Path(info).exists():
        daten = json.loads(Path(info).read_text(encoding="utf-8"))
        return daten.get("format", {}).get("identifier", {}).get("base-format", "revealjs")
    return "revealjs"


def _brand() -> dict:
    config = yaml.safe_load((ROOT / "_quarto.yml").read_text(encoding="utf-8"))
    return yaml.safe_load((ROOT / config["brand"]).read_text(encoding="utf-8"))


def _farben() -> dict:
    color = _brand()["color"]
    palette = color.get("palette", {})

    def aufloesen(wert: str) -> str:
        while wert in palette:
            wert = palette[wert]
        return wert

    return {name: aufloesen(wert) for name, wert in color.items() if isinstance(wert, str)}


def _ist_dunkel(hexfarbe: str) -> bool:
    r, g, b = mpl.colors.to_rgb(hexfarbe)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.5


FORMAT = _format()
FARBEN = _farben()
DUNKEL = _ist_dunkel(FARBEN.get("background", "#ffffff")) and FORMAT == "revealjs"

PRIMAER = FARBEN["primary"]
SEKUNDAER = FARBEN.get("secondary", "#6b6b6b")
if DUNKEL:
    TEXT = FARBEN.get("foreground", "#ffffff")
    GRAU = "#a0a0a0"
    HELLGRAU = "#3a3a3a"
else:
    TEXT = FARBEN.get("foreground", "#1d1d1f")
    TEXT = TEXT if _ist_dunkel(TEXT) else "#1d1d1f"
    GRAU = "#8a8a8a"
    HELLGRAU = "#e4e4e4"


def anwenden() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.transparent": DUNKEL,
            "figure.facecolor": "none" if DUNKEL else "white",
            "axes.facecolor": "none" if DUNKEL else "white",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.edgecolor": GRAU,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": HELLGRAU,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=[PRIMAER, SEKUNDAER, GRAU]),
            "svg.fonttype": "path",
        }
    )


anwenden()
