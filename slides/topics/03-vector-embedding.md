# Vector Embedding

----

### Was ist Vector Embedding?

- Umwandlung von Objekten (Wörter, Sätze, Bilder) in Vektoren in einem hochdimensionalen Raum
- Ziel: semantische Ähnlichkeiten und Beziehungen zwischen Objekten erfassen
- Grundlage für viele KI-Anwendungen, z.B. NLP, Bildverarbeitung, Empfehlungssystem
- Beispiel: Netflix, Spotify...

----

<img src="slides/img/embedding.png" alt="Picture" width="1000">

<p style="font-size:0.5em; margin-top:0.5em;">Quelle: <a href="https://medium.com/a-intelligence/understanding-text-to-vector-representation-from-tokenization-to-embeddings-b59dfdabbbae" target="_blank" rel="noopener">https://medium.com/a-intelligence/understanding-text-to-vector-representation-from-tokenization-to-embeddings-b59dfdabbbae</a></p>

##### Demo: TensorFlow Embedding Projector

<a href="https://projector.tensorflow.org/" target="_blank" rel="noopener">Demo</a>

----

### Ähnlichkeiten

- Vektoren können verglichen werden, um Ähnlichkeiten zu messen
- Metriken:
  - Euklidische Distanz (Abstand)
  - Manhattan-Distanz (Abstand)
  - Kosinus-Ähnlichkeit (Winkel zwischen Vektoren)
  - Skalarprodukt, wenn Vektoren normiert sind

----

### Zusammenhänge und Distanzen

- Mann zu Frau wie König zu _______?
- Vater zu Mutter wie Bruder zu _______?
- Paris zu Frankreich wie Berlin zu _______?

----

### Anwendung: Empfehlungssysteme

- **Netflix, Spotify, Amazon** nutzen diese Technik (Recommender Systems)
- Nicht nur Wörter, auch Nutzer und Inhalte (Filme, Songs) sind Vektoren
- **Content-Based:** Vektor des Nutzers ist nahe am Vektor des Films $\rightarrow$ Empfehlung
- **Collaborative:** Vektor von Nutzer A ist nahe am Vektor von Nutzer B $\rightarrow$ A bekommt Empfehlungen, die B mochte

----

### Anwendung: Semantische Suche

- **Klassische Suche:** Schlüsselwörter (Keywords)
  - Suche: "PKW kaufen" $\rightarrow$ findet Text "Auto zu verkaufen" eventuell nicht
  - Problem: Computer sucht nur exakt nach der Zeichenkette "P-K-W"
- **Vektorsuche:** Bedeutung (Semantik)
  - Suche: "Geld abheben" $\rightarrow$ findet "Bankomat", auch wenn das Wort nicht vorkommt
  - Grund: Die Vektoren von "Geld abheben" und "Bankomat" liegen im Vektorraum sehr nah beieinander
