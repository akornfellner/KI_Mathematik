"""Bindet alle Kapitel aus kapitel/ automatisch ein (läuft vor jedem Rendern).

Erzeugt _kapitel.qmd mit einer include-Zeile pro Datei kapitel/NN-name.qmd,
sortiert nach Dateiname. Dateien, die nicht mit einer Zahl beginnen
(z. B. _entwurf.qmd), werden ignoriert.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIEL = ROOT / "_kapitel.qmd"

kapitel = sorted(p for p in (ROOT / "kapitel").glob("*.qmd") if re.match(r"^\d", p.name))
inhalt = "<!-- Automatisch erzeugt von scripts/kapitel.py – nicht bearbeiten. -->\n\n" + "".join(
    f"{{{{< include kapitel/{p.name} >}}}}\n\n" for p in kapitel
)

# nur schreiben, wenn sich etwas geändert hat (sonst rendert die Vorschau endlos neu)
if not ZIEL.exists() or ZIEL.read_text(encoding="utf-8") != inhalt:
    ZIEL.write_text(inhalt, encoding="utf-8")
