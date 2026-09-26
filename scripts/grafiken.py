"""Grafiken für die Folien und das Handout (werden beim Rendern erzeugt)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from plotstil import GRAU, HELLGRAU, PRIMAER, SEKUNDAER, TEXT


def _knoten(ax, x, y, text, farbe=PRIMAER, r=0.42, textfarbe="white", groesse=15):
    ax.add_patch(Circle((x, y), r, facecolor=farbe, edgecolor="white", linewidth=2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", color=textfarbe, fontsize=groesse, zorder=4)


def _pfeil(ax, start, ende, farbe=GRAU, r=0.42, **kwargs):
    (x0, y0), (x1, y1) = start, ende
    d = np.hypot(x1 - x0, y1 - y0)
    dx, dy = (x1 - x0) / d, (y1 - y0) / d
    ax.add_patch(
        FancyArrowPatch(
            (x0 + dx * r, y0 + dy * r),
            (x1 - dx * r, y1 - dy * r),
            arrowstyle="-|>",
            mutation_scale=16,
            color=farbe,
            linewidth=2,
            zorder=2,
            **kwargs,
        )
    )


def netz_ausschnitt(zahlen: bool = False):
    """Ausgabeneuron o mit zwei Eingängen h1, h2 – für das Kettenregel-Beispiel."""
    if zahlen:
        h1, h2, w1, w2, b, o, t = "0{,}4", "0{,}6", "0{,}5", "0{,}3", "0", "0{,}5939", "1"
    else:
        h1, h2, w1, w2, b, o, t = "$h_1$", "$h_2$", "$w_1$", "$w_2$", "$b$", "$o$", "$t$"

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    p_h1, p_h2, p_o = (0.6, 2.4), (0.6, 0.4), (4.2, 1.4)

    _pfeil(ax, p_h1, p_o)
    _pfeil(ax, p_h2, p_o)
    _knoten(ax, *p_h1, "$h_1$", farbe=SEKUNDAER)
    _knoten(ax, *p_h2, "$h_2$", farbe=SEKUNDAER)
    _knoten(ax, *p_o, "$o$", r=0.5)

    ax.text(2.35, 2.25, f"$w_1 = {w1}$" if zahlen else w1, ha="center", fontsize=15)
    ax.text(2.35, 0.35, f"$w_2 = {w2}$" if zahlen else w2, ha="center", fontsize=15)
    ax.text(4.2, 0.55, f"$b = {b}$" if zahlen else f"Bias {b}", ha="center", va="top", fontsize=14)

    if zahlen:
        ax.text(0.0, 2.4, f"${h1}$", ha="right", va="center", fontsize=15)
        ax.text(0.0, 0.4, f"${h2}$", ha="right", va="center", fontsize=15)
        ax.text(4.85, 1.4, f"$o = {o}$", ha="left", va="center", fontsize=15, color=PRIMAER)
        ax.text(4.85, 0.85, f"Sollwert $t = {t}$", ha="left", va="center", fontsize=14, color=GRAU)
    else:
        ax.text(4.85, 1.4, "$o = \\sigma(w_1 h_1 + w_2 h_2 + b)$", ha="left", va="center", fontsize=14)
        ax.text(4.85, 0.85, f"Sollwert {t}", ha="left", va="center", fontsize=14, color=GRAU)

    ax.set_xlim(-0.8, 8.6)
    ax.set_ylim(-0.3, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def temperatur():
    """Wahrscheinlichkeiten für das nächste Wort bei verschiedenen Temperaturen (Softmax)."""
    woerter = ["blau", "grün", "rot", "schwarz", "gelb", "lila", "Elefant"]
    logits = np.array([3.0, 2.2, 1.9, 1.2, 0.9, 0.4, -2.5])
    temperaturen = [0.2, 1.0, 2.0]

    fig, achsen = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, T in zip(achsen, temperaturen):
        z = logits / T
        p = np.exp(z - z.max())
        p /= p.sum()
        ax.bar(woerter, p, color=PRIMAER, width=0.7)
        ax.set_title(f"Temperatur $T = {str(T).replace('.', '{,}')}$", color=TEXT)
        ax.tick_params(axis="x", rotation=45, labelsize=11)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True)
        for i, wert in enumerate(p):
            if wert >= 0.005:
                ax.text(i, wert + 0.02, f"{wert:.0%}", ha="center", fontsize=9, color=TEXT)
    achsen[0].set_ylabel("Wahrscheinlichkeit")
    achsen[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle("„Meine Lieblingsfarbe ist …“", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def wortvektoren():
    """Vereinfachte 2D-Darstellung: König − Mann + Frau ≈ Königin."""
    punkte = {
        "Mann": (1.0, 1.0),
        "Frau": (1.6, 3.0),
        "König": (4.2, 1.6),
        "Königin": (4.8, 3.6),
    }
    fig, ax = plt.subplots(figsize=(7, 4.4))

    def pfeil(a, b, farbe, stil="-"):
        ax.annotate(
            "",
            xy=punkte[b],
            xytext=punkte[a],
            arrowprops=dict(arrowstyle="-|>", color=farbe, lw=2.2, linestyle=stil, mutation_scale=16),
        )

    pfeil("Mann", "Frau", SEKUNDAER)
    pfeil("König", "Königin", SEKUNDAER, "--")
    pfeil("Mann", "König", GRAU)
    pfeil("Frau", "Königin", GRAU, "--")

    for name, (x, y) in punkte.items():
        ax.scatter(x, y, s=90, color=PRIMAER, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(x + 0.12, y + 0.12, name, fontsize=14)

    ax.text(0.55, 2.15, "„weiblich“", color=SEKUNDAER, fontsize=12, rotation=70)
    ax.text(2.5, 1.0, "„königlich“", color=GRAU, fontsize=12, rotation=8)

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4.4)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("König − Mann + Frau ≈ Königin", fontsize=14)
    return fig


def agent_schleife():
    """Der Arbeitszyklus eines KI-Agenten."""
    schritte = [
        ("Verstehen", "Ziel klären,\nRückfragen stellen"),
        ("Kontext", "Dateien lesen,\nrecherchieren"),
        ("Planen", "Schritte\nfestlegen"),
        ("Handeln", "Dateien schreiben,\nBefehle ausführen"),
        ("Prüfen", "kompilieren,\nnachrechnen"),
    ]
    n = len(schritte)
    fig, ax = plt.subplots(figsize=(9, 6.6))
    R, r = 2.2, 0.8
    winkel = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    pos = [(R * np.cos(w), R * np.sin(w)) for w in winkel]

    for i in range(n):
        ax.add_patch(
            FancyArrowPatch(
                pos[i],
                pos[(i + 1) % n],
                connectionstyle="arc3,rad=-0.28",
                arrowstyle="-|>",
                mutation_scale=20,
                color=GRAU,
                linewidth=2,
                shrinkA=40,
                shrinkB=40,
                zorder=1,
            )
        )

    for (titel, detail), (x, y), w in zip(schritte, pos, winkel):
        ax.add_patch(Circle((x, y), r, facecolor=PRIMAER, edgecolor="white", linewidth=3, zorder=2))
        ax.text(x, y, titel, ha="center", va="center", color="white", fontsize=11.5, weight="bold", zorder=3)
        cx, cy = np.cos(w), np.sin(w)
        ha = "center" if abs(cx) < 0.3 else ("left" if cx > 0 else "right")
        va = "bottom" if cy > 0.9 else ("center" if abs(cy) < 0.9 else "top")
        ax.text(
            x + cx * (r + 0.18),
            y + cy * (r + 0.18),
            detail,
            ha=ha,
            va=va,
            fontsize=11,
            color=TEXT,
            linespacing=1.3,
        )

    ax.add_patch(
        FancyBboxPatch(
            (-1.15, -0.42), 2.3, 0.84, boxstyle="round,pad=0.08", facecolor=HELLGRAU, edgecolor="none", zorder=0
        )
    )
    ax.text(0, 0.1, "Mensch", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0, -0.2, "gibt Ziel vor & prüft", ha="center", va="center", fontsize=10, color=GRAU)

    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-3.3, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig
