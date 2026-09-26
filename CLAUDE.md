# CLAUDE.md

Präsentation „KI und Mathematik“ von Alexander Kornfellner für **Mathematik-Lehrkräfte** (eingesetzt an HAK Eferding und HTL Leonding).
Aus **einer Quelle** entstehen Reveal.js-Folien (`_site/index.html`) und ein Handout (`_site/KI_Mathematik_Handout.pdf`).
Inhalte und Kommunikation auf Deutsch. Aufgebaut nach der Vorlage `praes-temp` (ehemals `reveal-temp`).

Kapitel: 01 Einleitung/Was ist KI · 02 Neuronale Netze · 03 Vector Embedding · 04 LLMs · 05 Arbeiten mit KI-Agenten.
Kapitel 5 beschreibt aktuelle Werkzeuge (Claude Code, Codex, …) – vor Änderungen daran aktuelle Informationen recherchieren.

## Befehle

- `just preview` – Live-Vorschau der Folien · `just preview-handout` – Vorschau des PDFs
- `just render` – Folien und Handout nach `_site/` · `just handout` – nur das PDF
- `just design htl|hak|white|dark` – Design wechseln
- Python-Pakete: `uv add <paket>` (landet in `.venv`, `pyproject.toml`, `uv.lock`), nie `pip install`
- Nach Änderungen immer `just render` ausführen und prüfen, dass es ohne `ERROR`/`WARN` durchläuft.

## Aufbau

- `_quarto.yml` – **zentrale Einstellungen**: Block `vortrag:` (titel, untertitel, autor, url), Zeile `brand:` (Design), `lang`, Handout-Dateiname (`format.typst.output-file`)
- `kapitel/NN-name.qmd` – **die Inhalte**, eine Datei pro Kapitel; werden nach Dateiname sortiert automatisch eingebunden (`scripts/kapitel.py` erzeugt `_kapitel.qmd`). Dateien ohne Ziffer am Anfang werden ignoriert.
- `index.qmd` – Rahmen (Python-Setup + Include); normalerweise nicht anfassen
- `brand/*.yml` + Logos – die vier Designs
- `filter/praesentation.lua` – Titelfolie, Handout-Titelseite, Kopfzeile, Inhaltsverzeichnis (aus `vortrag:` + Brand)
- `filter/handout.lua` – Anpassungen nur fürs PDF (siehe unten)
- `theme/slides.scss` – Folien-Styles · `theme/handout.typ` – PDF-Styles
- `scripts/plotstil.py` – matplotlib-Stil und Farben aus dem Design · `scripts/grafiken.py` – Grafiken dieser Präsentation (Netz-Ausschnitt, Temperatur, Wortvektoren, Agent-Schleife)
- `img/` – Bilder · `.github/workflows/build.yml` – Deployment auf GitHub Pages

## Designs

| brand             | Aussehen                             | Logo             |
|-------------------|--------------------------------------|------------------|
| `brand/htl.yml`   | weiß, Überschriften blau `#1a5276`   | HTL Leonding     |
| `brand/hak.yml`   | weiß, Überschriften rot `#c1272d`    | HAK/HAS Eferding |
| `brand/white.yml` | weiß, neutral, Akzent `#2a76dd`      | –                |
| `brand/dark.yml`  | dunkel `#191919`, Akzent `#42affa`   | –                |

Das Handout ist **immer weiß** (auch bei `dark`). Grafiken bekommen bei `dark` auf den Folien helle Schrift
und transparenten Hintergrund, im Handout dunkle Schrift – das regelt `plotstil.py` automatisch.

## Kapitel schreiben

- `# Titel` = Kapitel (Folien: neuer Abschnitt, Handout: nummeriert „1.“, „2.“ …)
- `## Titel` = Folie (Handout: unnummerierte Unterüberschrift)
- `## Titel {.slide-only}` = Überschrift nur auf den Folien; im Handout läuft der Inhalt unter der vorigen Überschrift weiter (für Folgefolien, reine Bildfolien, Demo-Folien)
- `---` = neue Folie ohne Titel (im Handout entfernt)
- `::: notes` = Sprechernotizen (Folien: nur Sprecheransicht, Taste S); **im Handout normaler Fließtext**. Hier stehen die Erklärungen, die mündlich ergänzt werden – in ganzen Sätzen schreiben.
- `::: quelle` = Quellenangabe (klein und dezent)
- Callouts für Wichtiges: `.callout-note` (mit `## Definition`), `.callout-tip`, `.callout-important`, `.callout-warning`, `.callout-caution`
- Nur Folien / nur Handout: `::: {.content-visible when-format="revealjs"}` bzw. `when-format="typst"`
- Bilder: `![](img/datei.png){fig-align="center" width="60%"}`
- Formeln in LaTeX, Dezimalkomma immer als `{,}` schreiben (`0{,}5`)
- Folien knapp halten (Stichpunkte), Erklärungen in `notes`

## Grafiken

- Neue Grafiken **immer mit matplotlib** erzeugen, nicht als fertige Bilder.
- Farben nur aus `plotstil` (`PRIMAER`, `SEKUNDAER`, `TEXT`, `GRAU`, `HELLGRAU`), nie fest codieren – sonst passen sie nicht zu allen Designs.
- Längere Grafiken als Funktion in `scripts/grafiken.py` (gibt `fig` zurück), im Kapitel aufrufen:

  ````markdown
  ```{python}
  #| fig-align: center
  #| out-width: 60%
  grafiken.meine_grafik();
  ```
  ````

- `plotstil` und `grafiken` sind in `index.qmd` bereits importiert. In Beschriftungen mit `$…$` ebenfalls `{,}` fürs Dezimalkomma.

## Stolperfallen (bereits gelöst – nicht rückgängig machen)

- `@` in Linktexten wird als Zitat gelesen → `\@` schreiben (sonst Typst-Fehler „does not contain a bibliography“).
- `execute: daemon: false` ist nötig, damit `plotstil` erkennt, ob gerade Folien oder Handout gerendert werden.
- Schriftname „Source Sans 3“ muss in CSS in Anführungszeichen stehen → in `theme/slides.scss` über `--r-main-font` gesetzt.
- `handout.lua` wandelt „…“ in Typst-Anführungszeichen um und entfernt `\left`/`\right` in Formeln; `handout.typ` setzt das Dezimalkomma ohne Abstand.
- Pagetitle und Fußzeile kommen über YAML-Anker (`&titel`, `&autor`) aus dem `vortrag:`-Block; `author` nur beim revealjs-Format setzen (sonst leere Autorenseite im PDF).
- `_kapitel.qmd` ist eingecheckt, weil `quarto preview` Includes vor dem Pre-Render auflöst. Neue Kapitel erscheinen nach Neustart der Vorschau.

## Deployment

Push auf `main` → GitHub Actions rendert **nur die Folien** und veröffentlicht sie auf GitHub Pages
(Handout bleibt lokal). Im Repo muss unter *Settings → Pages* die Quelle „GitHub Actions“ gewählt sein.
