# Prompting

----

### Prompt Engineering

- **Prompt**: Eingabeaufforderung oder Frage an eine Generative KI
- **Prompt Engineering**: Kunst und Wissenschaft, effektive Prompts zu erstellen, um gewünschte Ergebnisse von KI-Modellen zu erhalten

----

### Prompting Techniken

- Es gibt viele Techniken, um bessere Prompts zu erstellen
- nicht für alle Aufgaben gleich gut geeignet
- viele davon sind logisch und intuitiv

----

### Präzise Sprache

- Chatbots sind keine Suchmaschinen
- ganze Sätze und menschliche Sprache verwenden
- strukturierte Anfragen stellen
- Abkürzungen vermeiden
- nichts voraussetzen

----

### Rolle zuweisen

- Weise dem Modell eine Rolle zu, um den Kontext zu setzen
  - Ansatz 1 - Experte: "Du bist ein erfahrener Mathematiklehrer..."
  - Ansatz 2 - Charakter: "Du bist Albert Einstein..."
  - Ansatz 3 - Zielgruppe: "Gib mir Rückmeldung aus der Sicht eines 16 jährigen Schülers der Handelsakademie"

----

### Kontext bereitstellen

- KI kann keine Gedanken lesen
- so viele relevante Informationen wie möglich bereitstellen
- Orientierung an W-Fragen möglich: Wer? Was? Wann? Wo? Warum? Wie?
- Beispiel: "Erstelle eine 4-wöchige Unterrichtssequenz für den Mathe-Unterricht zum Thema Integralrechnung.” Zu viel Spielraum: Wie viele Stunden stehen tatsächlich zur Verfügung. 4 x 1 x 45 Minuten, oder 4 x 3 x 50 Minuten?

----

### Beispiele geben

- Beispiele helfen dem Modell, den gewünschten Stil und das Format zu verstehen
- Beispiel: "Erstelle ein Arbeitsblatt zum Thema XY für mich. Achte bei Komplexität und Umfang auf folgendes Beispiel, ändere aber die Angabe"

----

### Retrieval Augmented Generation (RAG)

- Kombiniert LLMs mit externen Wissensquellen
- Modell greift auf Dokumente, Datenbanken oder das Internet zu, um Antworten zu generieren
- Vorteile:
  - Aktuelles Wissen
  - Reduzierte Halluzinationen
- Beispiel: Ehemalige Zentralmatura-PDFs als Wissensbasis hochladen um bei der Erstellung neuer Aufgaben im typischen Stil zu bleiben

----

### Rückfragen

- Fordere das Modell auf, Rückfragen zu stellen, wenn Informationen fehlen
- Beispiel: "Erstelle eine Stundenwiederholung zum Thema XY. Frage bitte zuerst nach allen Informationen die du brauchst, damit du mir bestmöglich helfen kannst."

----

### Vorlagen erstellen

- Erstelle Vorlagen für wiederkehrende Aufgaben
- Wenn ein Prompt gut funktioniert, speichere ihn für zukünftige Verwendung

----

### Nach dem Prompt

- Überprüfe die Antwort kritisch
- Fordere bei Bedarf Verbesserungen an
- Iteratives Vorgehen: "Verbessere die Antwort, indem du ... berücksichtigst"

----

### System Prompts

- Einige Plattformen erlauben spezielle System Prompts
- Setzen den Rahmen für alle folgenden Interaktionen
- Beispiel: "Du bist ein hilfsbereiter Assistent, der immer höflich und professionell antwortet."
- In den User Einstellungen von ChatGPT können eigene System Prompts definiert werden
- Custom GPTs
- *Demo*: Custom GPTs

----

### Tipps

- Experimentiere mit verschiedenen Formulierungen
- Prompting ist Übungssache
- [Markdown](https://www.markdownguide.org/) oder **LaTeX** verwenden
- **Mathematik spezifisch**:
  - Analysetool zum Rechnen verwenden (lassen)
  - Matplotlib für Grafiken verwenden
