// Handout-Anpassungen (werden am Anfang des Dokuments eingebunden).
// Titelseite, Kopfzeile und Inhaltsverzeichnis erzeugt filter/praesentation.lua.

// Kein Logo im Seitenhintergrund (Quarto setzt es sonst auf jede Seite)
#set page(background: none)

// Nur Kapitel nummerieren (1., 2., …), Unterabschnitte ohne Nummer
#set heading(numbering: (..n) => if n.pos().len() == 1 { numbering("1.", ..n) })

#set par(justify: true, leading: 0.7em)
#set text(hyphenate: true)
#show table: set par(justify: false)

// Dezimalkomma in Formeln ohne Abstand: 0,5 statt 0, 5
#show math.equation: it => {
  show ",": math.class("normal", ",")
  it
}

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(1em)
  set text(size: 1.5em)
  it
  v(0.4em)
}
#show heading.where(level: 2): set block(above: 1.6em, below: 0.8em)
#show figure: set block(above: 1.2em, below: 1.2em)
