# Johnson – Umgewichten statt neu erfinden – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-johnson-demo.streamlit.app/)**

Siebtes Stück der **Kürzeste-Wege-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", **Konvergenz** von [Bellman-Ford](../bellman-ford-demo) und [Floyd-Warshall](../floyd-warshall-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Johnson** – an einem wachsenden Beispiel.
**Einmal Bellman-Ford** (ab einem Hilfsknoten *q*) berechnet **Potenziale** *h*; mit ihnen wird jede Kante umgewichtet, *c′(u,v) = c(u,v) + h(u) − h(v) ≥ 0*. Danach darf **n-mal Dijkstra** laufen, und das Ergebnis wird zurückgerechnet: *d(s,t) = d′(s,t) − h(s) + h(t)*.
Der Trick: jede Route von *s* nach *t* ändert ihre Kosten um **dieselbe Konstante** *h(s) − h(t)* – die Potenziale der Zwischenknoten heben sich auf, die kürzesten Routen bleiben dieselben.

**Einordnung in die Reihe (die Kanten des Graphen):** Bellman-Ford zeigt, dass Potenziale die Routen nicht ändern; Floyd-Warshall zeigt, dass *n*³ in dünnen Netzen verliert. Johnson setzt beides zusammen.
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)                                       [gebaut]
  └─ dijkstra-demo (Kosten korrekt, blind in alle Richtungen)                        [gebaut]
       ├─ bidirectional-demo → contraction-hierarchies-demo                          [gebaut]
       ├─ bellman-ford-demo ─┐                                                       [gebaut]
       │   floyd-warshall-demo ─┴→ johnson-demo (Konvergenz: Umgewichtung)           [dieses Stück]
       └─ multicriteria-demo (Zeit gegen CO₂, Pareto)                                [gebaut]
```

## Quellen

| Bestandteil | Quelle |
|---|---|
| Verfahren (Hilfsknoten, Potenziale aus Bellman-Ford, Umgewichten, n-mal Dijkstra, Zurückrechnen) | Johnson (1977); die Bücher (*Grokking Algorithms*, *Optimization Algorithms*) behandeln Johnson nicht, ein Buchbeispiel gibt es nicht zu spiegeln |
| Umsetzung, die "Falle" (alle Kosten anheben), Bildfolge, Zähler | eigen (Dijkstra, Bellman-Ford und Floyd-Warshall aus den Vorgänger-Demos) |
| Alle Netze | **eigene Graphen und Erzeuger**: kleines Liefernetz (mit einer negativen Kante, und die Variante mit negativem Zyklus), E-Lieferwagen mit Rekuperation im hügeligen Stadtnetz, Zufallsnetz mit Potenzialen, Stadtnetz ohne negative Kanten |
| Zahlen | **eigene Messungen** an diesen Netzen |

Aus den Büchern stammt keine Zahl, kein Graph und kein Text. **Kein OpenStreetMap-Auszug**, also keine ODbL-Pflichten.

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Kleines Netz (6 Orte, eine Rückvergütung Ost → Nord) | ✅ Bellman-Ford ab *q*: 3 Runden, **45** Kantenprüfungen, genau **ein** Potenzial ungleich null (Nord: −3); danach ist keine Kante negativ, n-mal Dijkstra braucht 54: zusammen **99** gegen **216** Vergleiche (Floyd-Warshall) und 180 (n-mal Bellman-Ford). n-mal Dijkstra ohne Umgewichten liegt an **7 von 30** Paaren falsch |
| Negativer Zyklus (Süd ↔ West: −2 Euro) | 🔁 Bellman-Ford ab *q* meldet ihn nach 2 Runden (**30** Kantenprüfungen); ohne Potenziale bricht Johnson ab |
| E-Lieferwagen (8 × 8, 30 m, 60 %) | ✅ **42** von 336 Kanten negativ, danach 0 (32 eng); Johnson **22 704** Schritte gegen **262 144** (Floyd-Warshall) und **137 424** (n-mal Bellman-Ford); n-mal Dijkstra ohne Umgewichten liegt an **524 von 4 032** Paaren falsch |
| Zufallsnetz (100 Knoten, Grad 3, Spanne 8) | ✅ **38** von 300 Kanten negativ; Johnson **31 600** (1 600 Bellman-Ford ab *q* + 30 000 Dijkstra) gegen 1 000 000 und 190 500 |
| Stadtnetz ohne negative Kanten (8 × 8) | ℹ️ alle Potenziale 0; die Sicherheit kostet **800** Kantenprüfungen (2 Runden), **3.7 %** der 21 504 des n-mal Dijkstra |
| Aufwand gegen Größe (Zufallsnetze, Grad 3, Spanne 8) | ✅ bei 400 Knoten **487 040** Schritte gegen **64 000 000** (Floyd-Warshall, 131-fach) und **3 891 840** (n-mal Bellman-Ford, 8-fach); der einmalige Bellman-Ford macht davon nur **1.4 %** aus |
| Aufwand gegen Dichte | ⚠️ im **vollständigen Netz** (60 Knoten) kehrt es sich um: Johnson **226 800**, Floyd-Warshall **216 000** – dort lohnt sich Johnson nicht mehr |
| Wenige Startknoten (200 Knoten) | ✅ der einmalige Bellman-Ford ab *q* (3 360) kostet etwa so viel wie einer ab einem Start (3 840): Johnson ist bei *k* = 1 **gleichauf** (3 960 gegen 3 840), bei *k* = 2 **klar billiger** (4 560 gegen 7 440) und bei *k* = 100 **6.7-mal** billiger als *k*-mal Bellman-Ford |
| Die Falle: alle Kosten um |kleinster Wert| anheben | ❌ die gefundene Route ist bei **8.7 %** (1.6 % negative Kanten) bis **34 %** (28 %) der Paare nicht die billigste; sie liegt nie daneben, wenn die richtige Route eine Kante hat (**0 %**), bei 4 Kanten an **10 %**, ab 6 Kanten an **65 %** der Paare – Anheben bestraft jede Kante |
| Korrektheit | ✅ Johnson liefert auf jedem geprüften Netz exakt die Matrix von Floyd-Warshall und networkx und meldet genau dann einen negativen Zyklus, wenn networkx einen findet; alle umgewichteten Kosten sind ≥ 0; jede Route ändert ihre Kosten um *h(s) − h(t)*; jedes andere zulässige Potenzial (auch die versteckten des Zufallsnetz-Erzeugers) gibt dieselbe Matrix; jede ausgepackte Route existiert im Originalnetz |

Die Zähler (Kantenprüfungen, Vergleiche) sind Schritte des Verfahrens und plattformfest, aber **kein gemeinsames Maß** (ein Vergleich in Floyd-Warshalls Matrixoperation ist billiger als eine Kantenprüfung mit Warteschlange). Laufzeiten stehen in der App nur als Messwerte und werden nirgends behauptet oder getestet:
Floyd-Warshall läuft vektorisiert (in C), Johnson und n-mal Bellman-Ford sind reines Python – die Laufzeiten sind deshalb nicht als Vergleich der Verfahren zu lesen.

## Was die Demo zeigt

1. **Johnson in Aktion** (Bildfolge-Regler + Abspielen): *Ausgangsnetz → Bellman-Ford ab q (Runde für Runde, das vorläufige Potenzial je Knoten) → umgewichtetes Netz (Kosten als "c′ (c)", enge Kanten gelb) → Dijkstra ab dem Start (bei großen Netzen etwa 40 Zwischenstände) → Ergebnis (zurückgerechnet, mit der Formel für das gezeigte Paar)*. Kleine Netze mit Tabelle (Ort | Potenzial | *d′ → d*), große mit Farbe; bei einem negativen Zyklus endet die Folge mit dem roten Zyklus.
2. **Umgewichten – und was es spart:** Zähler von Johnson (aufgeteilt in Bellman-Ford und Dijkstra), Floyd-Warshall und n-mal Bellman-Ford; negative Kanten vorher → nachher; enge Kanten; Urteil (✅ / ℹ️ / 🔁 / ⚠️ bei der Falle).
3. **Vergleich** (Expander) mit Laufzeiten als Messwerte; **Experimente auf Knopfdruck**: Aufwand gegen Dichte und Größe, wenige Startknoten, die Falle (nach Zahl der negativen Kanten und nach Kantenzahl der Route).
4. **Wo die Annahmen enden** (Tabelle) und **Mathematische Formulierung** (Potenzial-Lemma als Teleskopsumme, Nicht-Eindeutigkeit, warum "alle Kosten anheben" falsch ist, Aufwand); Verwandtes aus der Literatur (A\* mit konsistenter Heuristik, Ungarische Methode) ist als solches gekennzeichnet, in dieser Demo nicht gebaut (A* und Ungarische Methode gibt es als eigene Demos im Portfolio).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz, **Umgewichtung** (Johnson oder die Falle) und – bei erzeugten Netzen – das gezeigte Paar wählen; die Adresszeile spiegelt die Konfiguration (Permalink). Regler, die zum gewählten Netz nicht gehören, sind ausgeblendet. Höchstens 400 Knoten.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `jo_graph.py`, `jo_queues.py`, `jo_sp.py` | Graph in CSR-Form, Warteschlange, Dijkstra und Bellman-Ford (aus den Vorgänger-Demos) |
| `jo_fw.py` | Floyd-Warshall als Vergleich und Exaktheitsprobe |
| `jo_algorithm.py` | Hilfsknoten, Potenziale, Umgewichten, Dijkstra ab jedem Start, Zurückrechnen, die Falle |
| `jo_scenario.py` | Netze: kleines Netz (mit und ohne negativen Zyklus), E-Lieferwagen, Zufallsnetz, Stadtnetz |
| `jo_evaluation.py` | Kennzahlen, Paarwahl, Bildfolge, Experimente |
| `jo_visualization.py`, `jo_presets.py`, `jo_constants.py` | Abbildungen, Presets und Permalink, Konstanten |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe läuft gegen networkx (`floyd_warshall_numpy`, `negative_edge_cycle`) und Floyd-Warshall auf Netzen mit und ohne negative Kanten und Zyklen, mit unerreichbaren Paaren, Nullkanten und einem einzelnen Knoten; ein Regressionstest klickt "▶️ Abspielen" auf Netzen mit mehreren Bildern.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kürzeste Wege: von der Breitensuche bis RAPTOR](https://sebastianhanisch.net/konzepte-kuerzeste-wege.html).
