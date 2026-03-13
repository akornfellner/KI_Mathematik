# Neuronale Netze

----

### Anwendungen

- Bild- und Spracherkennung
- Natürliche Sprachverarbeitung (NLP)
- Empfehlungssysteme
- LLMs (z.B. ChatGPT)
- ...

----

### Inspiration

- Biologische Neuronen im menschlichen Gehirn

<img src="slides/img/neuron.jpeg" alt="Picture" width="1500">

----

### Künstliches Neuron

<img src="slides/img/neuron2.png" alt="Picture" width="800">

<p style="font-size:0.5em; margin-top:0.5em;">Quelle: <a href="https://methpsy.elearning.psych.tu-dresden.de/mediawiki/index.php/Neuronale_Netze" target="_blank" rel="noopener">TU Dresden</a></p>

----

### Mathematik im Neuron

- Gewichtete Summe
- Aktivierungsfunktion
- Bias (Schwellenwert)
- nicht **linear**

$$
\text{Output} = \sigma\left(\sum_{i=1}^{n} w_i \cdot x_i+b\right)
$$

----

### Aktivierungsfunktion

- durch Aktivierungsfunktionen kann Linearität verhindert werden
- bringt meinen Output in ein vorgegebenes Intervall
- Es gibt verschiedene:
  - Sigmoid
  - ReLU
  - Softmax
  - ...

----

#### Beispiel Sigmoid Funktion

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

<img src="slides/img/sigmoid.png" alt="Picture" width="450">

----

### Aufbau eines Neuronalen Netzes

- besteht aus mehreren Schichten (Layers)
- Kanten verbinden die Neuronen der Schichten und haben Gewichte
- Jede Schicht besteht aus mehreren Neuronen

----

### Aufbau eines Neuronalen Netzes

<img src="slides/img/nn.png" alt="Picture" width="800">

<p style="font-size:0.5em; margin-top:0.5em;">Quelle: <a href="https://rocketloop.de/de/blog/kuenstliche-neuronale-netze/" target="_blank" rel="noopener">https://rocketloop.de/de/blog/kuenstliche-neuronale-netze/</a></p>

----

### Beispiel zum händischen Rechnen

<img src="slides/img/calc_nn.png" alt="Picture" width="900">

*Hinweis: Gozintograph*

----

### Lösung

- $h1 = \sigma\left(0{,}8\cdot 0{,}6 + (-0{,}2) \cdot 0{,}4\right)= \frac{1}{1+e^{-0{,}4}}=0{,}5987$
- $h2 = \sigma\left(0{,}4\cdot 0{,}6 + 0{,}9 \cdot 0{,}4\right)= \frac{1}{1+e^{-0{,}6}}=0{,}6457$
- $y = \sigma\left(0{,}7 \cdot 0{,}5987 + (-0{,}5) \cdot 0{,}6457\right)=\mathbf{0{,}524}$

----

### Elegantere Schreibweise mit Matrizen

$$
\mathbf{h} = \sigma\left(\mathbf{W_1} \cdot \mathbf{x}+\mathbf{b}\right)
$$

- Demo Matrizenrechnung

----

### Vorteile Matrizenrechnung

- kompakter
- effizienter zu berechnen (v.a. bei großen Netzen)
- einfacher zu implementieren
- parallelisierbar (GPU)

----

### Anmerkungen zu Neuronalen Netzen

- Gewichte werden zufällig (meist normalverteilt) initialisiert
- Trainingsprozess passt die Gewichte an
- Neuronales Netz ist eine große, komplexe, nicht-lineare Funktion, die von vielen Variablen (Gewichte) abhängt
- GPT-3 hat z.B. ca. 175 Milliarden Gewichte in 96 Schichten

----

#### Wie lernen Neuronale Netze (Training)?

- Trainingsdaten: Eingabedaten + erwartete Ausgabe (Label)
- Vorhersage der Ausgabe durch das Netz
- Berechnung des Fehlers (Loss Function)
- Anpassung der Gewichte durch Backpropagation und Optimierungsverfahren (z.B. Gradient Descent)

----

### Backpropagation

- Fehler wird rückwärts durch das Netz propagiert
- anteilige Anpassung der Gewichte basierend auf ihrem Beitrag zum Fehler
- kann wieder mit Matrizenrechnung effizient berechnet und dargestellt werden mit den transponierten Matrizen der Gewichte

----

### Backpropagation

- Fehlerfunktion, z.B. mittlere quadratische Abweichung (MSE): $L=(\text{Soll} - \text{Ist})^2$
- Wie trägt ein Gewicht $w$ zum Fehler bei?
- partiellen Ableitung der Fehlerfunktion nach $w$ berechnen: $\frac{\partial }{\partial w}$
- Gewicht anpassen: $w_{\text{neu}} = w_{\text{alt}} - \alpha \cdot \frac{\partial L}{\partial w}$
- $\alpha$ = Lernrate

----

### Gradient Descent

- Optimierungsalgorithmus zur Minimierung der Fehlerfunktion
- Gewichte werden in Richtung des steilsten Abstiegs (negativer Gradient) der Fehlerfunktion angepasst
- Iterativer Prozess: viele Durchläufe (Epochen) über die Trainingsdaten
- Lernrate $\alpha$ bestimmt die Schrittgröße bei der Anpassung der Gewichte

----

### Gradient Descent

<img src="slides/img/gradient.png" alt="Picture" width="800">

<p style="font-size:0.5em; margin-top:0.5em;">Quelle: <a href="https://medium.com/@jaleeladejumo/gradient-descent-from-scratch-batch-gradient-descent-stochastic-gradient-descent-and-mini-batch-def681187473" target="_blank" rel="noopener">https://medium.com/@jaleeladejumo/gradient-descent-from-scratch-batch-gradient-descent-stochastic-gradient-descent-and-mini-batch-def681187473</a></p>

----

### händisches Beispiel

<img src="slides/img/kette1.png" alt="Picture" width="300">

$L = (t-o)^2 \text{ und } o = \sigma(w_1 \cdot h_1 + w_2 \cdot h_2+b)$

$\Rightarrow L = (t - \sigma(w_1 \cdot h_1 + w_2 \cdot h_2+b))^2$

$\mathbf{\frac{\partial L}{\partial w_1} = -2(t-o) \cdot o \cdot (1 - o) \cdot h_1}$

----

#### konkrete Zahlen

<img src="img/kette2.png" alt="Picture" width="600">

$\alpha = 0{,}1$

$w=0{,}5+0.1\cdot (1-0{,}5938)\cdot 0{,}5938\cdot (1-0{,}5938)\cdot 0{,}4$
$w=0{,}5039$

----

### Moderne Netzarten

- Convolutional Neural Networks (CNNs) — Bildverarbeitung
- Recurrent Neural Networks (RNNs) — Sequenzdaten
- Long Short-Term Memory (LSTM) — Langzeitabhängigkeiten
- Transformer — NLP, LLMs (z.B. GPT-4, Gemini)

----

### Mathematik in Neuronalen Netzen

- Lineare Algebra (Matrizenrechnung)
- Analysis (Ableitungen, Kettenregel, Aktivierungsfunktionen)
- Optimierung (Gradient Descent)
- Normalverteilung (Gewichtinitialisierung, Skalierung)

----

## Demo Time with MNIST!