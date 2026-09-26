default:
  @just --list

# Live-Vorschau der Folien im Browser
preview: kapitel
  uv run quarto preview --to revealjs

# Live-Vorschau des Handouts (PDF)
preview-handout: kapitel
  uv run quarto preview --to typst

# Folien und Handout nach _site/ rendern
render: kapitel
  uv run quarto render

# Nur das Handout erzeugen (Dateiname: output-file in _quarto.yml)
handout: kapitel
  uv run quarto render --to typst

# Design wechseln: just design htl | hak | white | dark
design name:
  test -f brand/{{name}}.yml || (echo "Unbekanntes Design: {{name}} (htl, hak, white, dark)" && exit 1)
  sed -i 's|^brand: brand/.*\.yml|brand: brand/{{name}}.yml|' _quarto.yml
  @grep '^brand:' _quarto.yml

# Kapitelliste neu erzeugen (passiert bei den anderen Befehlen automatisch)
kapitel:
  @python3 scripts/kapitel.py
