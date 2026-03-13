# Large Language Models (LLMs)

----

### Was sind LLMs?

- Große neuronale Netze, die auf riesigen Textdatenmengen trainiert wurden
- Ziel: Verständnis und Generierung natürlicher Sprache
- Beispiele: ChatGPT, Gemini, Claude, Llama...

<a href="https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5" target="_blank" rel="noopener">Video</a>

----

### Wie funktionieren LLMs?

- Basieren auf Transformer-Architektur
- Lernen komplexe Sprachmuster, Grammatik und Kontext
- **Trainiert durch Vorhersage des nächsten Wortes in einem Satz**

*Meine Lieblingsfarbe ist _____*

----

### Frühere Textvervollständiger

- Beispiel Smartphone: Ich - bin - in - einer - Gruppe - mit - den - anderen - Kindern - die - sich - um - mich - gekümmert - hat - ...
- Problem: Kontext wird nicht gut erfasst, lange Abhängigkeiten schwer zu lernen
- LLMs können längere Kontexte besser verstehen und nutzen

----

### 🧠 Was ist ein Transformer?

Ein **Transformer** ist eine spezielle Art von KI-Modell, die besonders gut darin ist, **Sprache zu verstehen und zu erzeugen**. Seit 2017 bildet diese Architektur die Grundlage fast aller modernen LLMs.

----

## 🔑 Die wichtigsten Ideen dahinter

----

### Attention – das Herzstück

Stell dir vor, du liest einen Satz und möchtest verstehen, worauf sich ein bestimmtes Wort bezieht. Der Transformer macht das auch – mit einem Mechanismus namens **Attention**.

* Attention bedeutet: *„Achte mehr auf die wichtigen Wörter, weniger auf die unwichtigen.“*
* So kann das Modell Zusammenhänge erkennen, egal wie lang der Satz ist.

----

#### Beispiel:
Im Satz *„Das ist mein Lieblingslied von Queen.“* muss das Modell verstehen, dass *„Queen“* sich auf die Band bezieht, nicht auf eine Königin.

----

### Parallel statt nacheinander

* Frühere Modelle (RNNs, LSTMs) mussten Wörter **nacheinander** verarbeiten.
* Transformer können viele Wörter **parallel** betrachten → viel schneller & effizienter.

----

### 🚀 Warum Transformer so erfolgreich sind

* ✔ **Verstehen langen Kontextes** durch Attention
* ✔ **Skalierbar** (funktioniert gut mit riesigen Datenmengen und großen Modellen)
* ✔ **Parallelisierbar** → viel schneller trainierbar
* ✔ **Flexibel**: für Texte, Bilder, Audio, Code usw.

----

### Training von LLMs

- Es besteht aus einem tiefen neuronalen Netz mit Millionen bis Milliarden von Parametern
- Trainiert auf riesigen Textkorpora (Bücher, Artikel, Webseiten)
- Trainiert auf Bildern mit Bildbeschreibungen (Multimodal Models)
- Menschen helfen durch Korrekturen (Reinforcement Learning with Human Feedback, RLHF)

----

### Halluzinationen

- LLMs können falsche oder erfundene Informationen generieren
- Ursachen:
  - Unvollständige oder fehlerhafte Trainingsdaten
  - Veraltetes Wissen (Cutoff-Datum)
  - basiert auf Wahrscheinlichkeiten, nicht Fakten
- **Lösung**: Faktenprüfung, Internetzugang, Deep Research, RAG

----

### Unterschiedliche Antworten

- Warum geben LLMs unterschiedliche Antworten auf dieselbe Frage?
- Beispiel: "Vervollständige den Satz: Das Notensystem geht von 1 bis ..."

----

![](slides/img/noten2.png)

<p style="font-size:0.5em; margin-top:0.5em;">Quelle: <a href="https://bildungsinformatik.at/2024/collective-gpt/" target="_blank" rel="noopener">https://bildungsinformatik.at/2024/collective-gpt/</a></p>

----

![](slides/img/noten1.png)

<p style="font-size:0.5em; margin-top:0.5em;">Quelle: <a href="https://bildungsinformatik.at/2024/collective-gpt/" target="_blank" rel="noopener">https://bildungsinformatik.at/2024/collective-gpt/</a></p>

----

### Temperatur

- immer das wahrscheinlichste Wort auswählen
  - deterministisch, immer gleiche Antwort
  - kann zu langweiligen Antworten führen
- Temperatur regelt Zufälligkeit:
  - **0**: Keine = blau, blau, blau, blau, ...
  - **1**: Hohe Zufälligkeit = blau, grün, rot, gelb, ...
  - **2**: Sehr hohe Zufälligkeit = Elefant, Auto, grün, 42, jklö, ...
- Standardtemperatur meist zwischen 0,6 und 0,8

----

### Tokens

- LLMs verarbeiten keine Wörter, sondern **Tokens**
- Token können ganze Wörter, Wortteile oder einzelne Zeichen sein

<img src="slides/img/token.png" alt="Picture" width="800">

<p style="font-size:0.5em; margin-top:0.5em;">Quelle: <a href="https://platform.openai.com/tokenizer" target="_blank" rel="noopener">https://platform.openai.com/tokenizer</a></p>

----

### Play around with different LLMs

- LMArena: https://lmarena.ai/
- Duck.AI: https://duckduckgo.com

----

### Chatbots

- LLMs sind keine Chatbots
- Chatbots sind Anwendungen, die auf LLMs aufbauen
- Chatbots haben oft zusätzliche Funktionen:
  - Internetsuche
  - Wissensdatenbanken
  - Userinterface
  - ...
