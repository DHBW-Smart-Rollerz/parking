# Konzept: KI-basierte Parkplatzerkennung und Einparkplanung mit YOLO26n-Pose

Studienarbeit DHBW Smart Rollerz, Paket `parking` (ROS 2 Jazzy). Stand: 05.11.2026.

Dieses Dokument beschreibt das Gesamtkonzept: Anforderungen, Systemarchitektur, die Wahl der Bildansicht, den Vergleich der Modellkonzepte mit der Begründung für **YOLO26n-Pose** als Detektor, die Nachverarbeitung (Rechteck-Fit, Tracking), die Einparkplanung, das Trainings- und Annotationskonzept, die Evaluation, den aktuellen Stand mit Messergebnissen sowie offene Punkte.

Zur Abgrenzung:
- [INFO_Prototyp.md](INFO_Prototyp.md) beschreibt die Bedienung und den zeitlichen Ablauf der Umsetzung.
- [docs/workflow.md](docs/workflow.md) beschreibt den Arbeitsablauf von Rosbag bis Training.

> **Stand der Umsetzung:** Pipeline, Planung, ROS-Anbindung und Trainingsumgebung sind umgesetzt und getestet. Trainiert wurden bisher YOLO11n-Pose-Modelle. YOLO26n-Pose ist in der installierten Ultralytics-Version 8.4.193 verfügbar und wird durch eine Zeile im Trainings-Notebook aktiviert (Abschnitt 6.4). Messwerte mit YOLO26n liegen noch nicht vor.

---

## Inhalt

1. [Aufgabenstellung und Anforderungen](#1-aufgabenstellung-und-anforderungen)
2. [Randbedingungen aus dem Regelwerk und der Hardware](#2-randbedingungen-aus-dem-regelwerk-und-der-hardware)
3. [Systemarchitektur](#3-systemarchitektur)
4. [Bildansicht: Vogelperspektive, Perspektive oder erweiterte Vogelperspektive](#4-bildansicht)
5. [Vergleich der Modellkonzepte](#5-vergleich-der-modellkonzepte)
6. [Detektor: YOLO26n-Pose](#6-detektor-yolo26n-pose)
7. [Nachverarbeitung: Geometrie und Tracking](#7-nachverarbeitung-geometrie-und-tracking)
8. [Zielauswahl und Einparkplanung](#8-zielauswahl-und-einparkplanung)
9. [Schnittstellen](#9-schnittstellen)
10. [Daten- und Annotationskonzept](#10-daten--und-annotationskonzept)
11. [Trainingskonzept](#11-trainingskonzept)
12. [Evaluationskonzept](#12-evaluationskonzept)
13. [Aktueller Stand und Messergebnisse](#13-aktueller-stand-und-messergebnisse)
14. [Grenzen, Risiken und offene Punkte](#14-grenzen-risiken-und-offene-punkte)
15. [Roadmap](#15-roadmap)
16. [Literatur](#16-literatur)
17. [Anhang: Dateien, Konfiguration, Befehle](#17-anhang)

---

## 1. Aufgabenstellung und Anforderungen

Ein autonomes Miniaturfahrzeug (Maßstab ca. 1:10) soll auf der Wettbewerbsstrecke Parkplätze finden, bewerten und eine fahrbare Bahn hinein bestimmen.

### 1.1 Funktionale Anforderungen

| ID | Anforderung |
|---|---|
| F1 | Erkennung von Parkplätzen aus der Vogelperspektive: **90°-Plätze links**, **Parallelplätze rechts** |
| F2 | Klassifikation jedes Platzes: **frei, besetzt, gesperrt** |
| F3 | Bestimmung der **Lage** jedes Platzes (Mittelpunkt, Ausrichtung, Größe) im Fahrzeug- bzw. Weltkoordinatensystem in mm |
| F4 | Prüfung, ob das Fahrzeug **in den Platz passt** (statt reinem Flächenvergleich) |
| F5 | Bestimmung einer **Zielpose** und einer **fahrbaren Bahn** in den Platz |
| F6 | Übergabe der Bahn an die Regelung in einer Form, die **Vorwärts- und Rückwärtsfahrt** abbildet |
| F7 | Debug-Ausgabe (Bild/Video) zur Kontrolle von Erkennung und Bahn |
| F8 | Testbarkeit ohne Fahrzeug auf aufgezeichneten **Rosbags** |

### 1.2 Nicht-funktionale Anforderungen

| ID | Anforderung |
|---|---|
| N1 | Echtzeitfähig auf dem Fahrzeugrechner (NUC, CPU), Ziel ≥ 10 Hz für die Parkplatzsuche |
| N2 | Robust gegen teilweise sichtbare Plätze, Lichtwechsel und Bewegungsunschärfe |
| N3 | Wenig Trainingsdaten (Hunderte, nicht Zehntausende Bilder) |
| N4 | Nachvollziehbar und wissenschaftlich vergleichbar (Studienarbeit) |
| N5 | Konsistent mit den Projektrichtlinien (ROS 2, `SmartyNode`, ruff-Konfiguration des Workspaces) |

### 1.3 Grundidee

Die KI löst nur die Teile, die klassisch schwer sind: **Wo** ist ein Platz, und **welche Klasse** hat er? Alles, was sich aus bekannter Geometrie ableiten lässt, wird klassisch und erklärbar gelöst: die genaue Lage, die Passprüfung und die Bahn. So bleibt das System mit wenig Daten robust und jeder Schritt einzeln prüfbar.

---

## 2. Randbedingungen aus dem Regelwerk und der Hardware

### 2.1 CAuDri-Challenge 2026 (Abschnitte 5.4, 7.4, A.5)

| Thema | Regel |
|---|---|
| 90°-Plätze (5.4.2, A.5.2) | links, alle gleich groß: **350 mm breit, 500 mm tief**; Linien 18–20 mm; gesperrt durch Hindernis oder Sperrfläche (X); „gesperrt, wenn das Fahrzeug nicht vollständig hineinpasst“; Hindernisse 20–100 mm von der linken Fahrbahnmarkierung |
| Parallelplätze (5.4.1, A.5.1) | rechts, **300 mm tief**; **Länge variabel**, in Fahrtrichtung zunehmend, der letzte ≥ 700 mm; Begrenzung vorne/hinten durch Hindernisse oder Sperrflächen; Lücken < 400 mm sind keine Plätze |
| Parkzone (5.4) | gerade, eben, gestrichelte Mittellinie, Schild 300–500 mm vor der ersten Fläche |
| Wertung (7.4) | ein Versuch pro Runde, Blinker in Parkrichtung; gültig, wenn **alle Räder im Platz** stehen; vorwärts und rückwärts erlaubt; Kollision macht den Versuch ungültig |
| Fahrspur | 350–450 mm breit (Innenkanten) |

### 2.2 Fahrzeug und Sensorik

| Größe | Wert | Quelle |
|---|---|---|
| Radstand | 260 mm | altes Paket `parking_spot` |
| max. Lenkwinkel | 35° | altes Paket |
| min. Wenderadius Hinterachse R_min | 260 / tan 35° = **371 mm** | berechnet |
| Länge × Breite | 370 × 200 mm | Simulationsmodell, **nachmessen** |
| Überhang hinten | 70 mm | Simulationsmodell, **nachmessen** |
| Hinterachse im Fahrzeug-KS | x = −85 mm | altes Paket, **nachmessen** |
| Kamera | Allied Vision (VimbaX), Rohbild 2064 × 1544, mono8, ca. 30 Hz | Rosbags |
| Pose | `/pose_estimation/pose` (EKF aus IMU, Geschwindigkeit, Lenkwinkel), Position in mm, `orientation.z` = Gierwinkel | Paket `pose_estimation` |

### 2.3 Daten

Vorhanden sind 6 Rosbags vom 28.10.2026, jeweils 18–35 s, die **nur `/camera/image/raw`** enthalten und keine Pose:

| Bag | Inhalt | Verwendung |
|---|---|---|
| 13_32_18 | Vorbeifahrt | Training (v4 bzw. BEV v1) |
| 13_32_53 | Vorbeifahrt + zwei 90°-Manöver | Training (v5) |
| 13_36_43 | Vorbeifahrt | Training (v6) |
| 13_35_42 | 90°-Einparken links | Training (neu) |
| 13_37_32 | Parallel-Einparken rechts | Training (neu) |
| 13_38_33 | Vorbeifahrt an Parallelplätzen | **Testset, nie trainieren** |

---

## 3. Systemarchitektur

### 3.1 ROS-Graph

```text
vimbax_camera ──/camera/image/raw──► camera_preprocessing ──/camera/image/undistorted──┐
                                       (Entzerrung, BEV)  ──/camera/image/bev──────────┤
                                                                                        ▼
pose_estimation ──/pose_estimation/pose──────────────────────────────► parking_perception_node
                                                                         │
                       ┌─────────────────────────────────────────────────┤
                       ▼                  ▼                  ▼           ▼
               /parking/path   /parking/target_pose   /parking/path_segments   /parking/debug_image
               (nav_msgs/Path) (PoseStamped)          (κ(s), Float32MultiArray) (nur debug)
                       │                                     │
                       └──────────────► Regelung (Pure Pursuit / Stanley) ◄──┘   [nicht Teil des Prototyps]
```

### 3.2 Verarbeitungskette in der Node

```text
Bild (Topic der gewählten Ansicht)
  │  view.prepare()                     ── perception/view.py, perception/bev.py
  ▼
Eingangsbild des Detektors (z. B. Parking-BEV 800×717)
  │  YOLO26n-Pose                       ── perception/yolo_model.py
  ▼
Detektionen: Klasse, Score, 4 Ecken [px], Konfidenz je Ecke
  │  Pixel → mm (view.image_to_world)   ── perception/model.py
  ▼
ParkingSpot (Fahrzeug-KS, mm) + Sichtbarkeit je Ecke
  │  Rechteck-Fit mit Regelmaßen         ── perception/geometry.py
  │  Tracking über Frames                ── perception/tracker.py
  ▼
Tracks: Pose, Größe, Klasse (Mehrheitsentscheid), Bestätigung
  │  Prüfung + Zielpose + Bahn           ── planning/target.py, planning/planner.py
  ▼
Ziel + Bahn κ(s)                         ── planning/path.py
  │  Veröffentlichung, Debug-Bild        ── parking_perception_node.py, perception/visualization.py
  ▼
ROS-Topics
```

### 3.3 Schichtung und Designprinzipien

| Schicht | Module | ROS-abhängig | Zweck |
|---|---|---|---|
| ROS-Anbindung | `parking_perception_node.py` | ja | Parameter, Topics, Timer, Nachrichten |
| Ansicht | `perception/view.py`, `perception/bev.py` | nein (Kalibrierung aus `camera_preprocessing`) | Bildvorbereitung, px ↔ mm |
| Detektion | `perception/model.py`, `perception/yolo_model.py` | nein | Netz, Sichtbarkeit der Ecken |
| Geometrie | `perception/geometry.py` | nein | Pose2D, Kabsch, Rechteck-Fit, Polygon-Abstände |
| Tracking | `perception/tracker.py` | nein | Zuordnung, Glättung, Klassen-Vote |
| Planung | `planning/*` | nein | Fahrzeug, Prüfungen, Planer, Bahn |
| Orchestrierung | `pipeline.py` | nein | verbindet Geometrie, Tracking, Planung |
| Werkzeuge | `tools/*.py` | Rosbag-Lesen | Extraktion, Offline-Test, CVAT-Anbindung |

Prinzipien:
- **ROS-freier Kern:** Die Geometrie, das Tracking und die Planung sind reines numpy und mit pytest testbar (37 Unit-Tests). Node und Offline-Tool verwenden denselben Code.
- **Konfiguration statt Code:** Ansicht, Modell, Regelmaße, Fahrzeug, Tracking und Planung stehen in YAML-Dateien (Abschnitt 17.2).
- **Austauschbarer Detektor:** `detector: 'yolo_pose' | 'parking_net'` und ein Modell pro Ansicht.

---

## 4. Bildansicht

### 4.1 Optionen

| Ansicht | Beschreibung | Abdeckung Boden | Auflösung | Form der Plätze |
|---|---|---|---|---|
| `bev` (camera_preprocessing) | feste Homographie, 800 × 640 | x −0,12…1,28 m, y −0,92…1,0 m, ~2,0 m² | ~2,2 × 2,4 mm/px konstant | Rechtecke, konstante Größe |
| `undistorted` | entzerrtes Kamerabild, 800 × 640 | bis ≥ 3 m, ±2 m, ~8,3 m² (bis 3 m geprüft) | 3/1,3 mm/px bei 0,3 m; 12/2,7 bei 1 m; 50/5,5 bei 2,5 m (vor/seitlich) | Trapeze, Größe abhängig von der Entfernung |
| **`parking_bev`** (dieses Paket) | eigene Homographie aus dem entzerrten Bild | x −0,15…2,0 m, y ±1,2 m | 3 mm/px konstant, 800 × 717 | Rechtecke, konstante Größe |

### 4.2 Entscheidung: `parking_bev`

- **Gegen die alte BEV:** Sie zeigt nur etwa ein Viertel des sichtbaren Bodens. Von den in ihr abgeschnittenen Plätzen wären im entzerrten Bild 29–48 % vollständig sichtbar.
- **Gegen die Perspektive:** Ferne Plätze sind klein und in Fahrtrichtung kaum aufgelöst (1 px ≈ 12 mm bei 1 m, ≈ 50 mm bei 2,5 m). Das Netz müsste starke Größenänderungen lernen, was mit wenigen Daten schwer ist. Die Literatur zur Parkplatzerkennung (DeepPS, DMPR-PS, GCN) arbeitet bewusst in der Vogelperspektive, weil Plätze dort metrisch, rechteckig und gleich groß sind.
- **Für die eigene BEV:** größere Reichweite bei gleichen Vorteilen der Vogelperspektive. Die Umrechnung px ↔ mm ist eine lineare Abbildung (`pixel_to_world`, `world_to_pixel`). Die Konsistenz mit der Kalibrierung ist geprüft: Die Helligkeitskorrelation an gleichen Bodenpunkten liegt meist ≥ 0,99. Die BEV von `camera_preprocessing` bleibt für andere Nodes unverändert.

Grundsätzliche Grenzen jeder Ansicht:
- **Kein Informationsgewinn:** Die Vogelperspektive ist eine Umrechnung des entzerrten Bildes und enthält keine zusätzliche Information. Sie kann aber Information verlieren, wenn der Ausschnitt zu klein ist.
- **Toter Winkel neben dem Fahrzeug:** Die schwarzen Bereiche unten links und rechts liegen außerhalb des Kamerasichtfelds. Diesen Boden sieht keine Ansicht, die Lücke überbrückt das Tracking.
- **Fernbereich:** Oberhalb von ~1,3 m ist das Bild unscharf, weil es aus wenigen echten Pixeln hochgerechnet ist.
- **Bekannte Schwäche der aktuellen Wahl:** Die 90°-Plätze reichen bis y ≈ 1,27–1,34 m, die Ansicht endet bei 1,2 m. Ihre äußeren Ecken liegen deshalb **immer außerhalb des Bildes** (Abschnitt 14).

### 4.3 Binarisierte Eingabe

Eine Binarisierung mit festem Schwellwert, wie im alten Paket (`THRESH_VALUE = 65`), wurde verworfen:
- **Hindernisse verschmelzen mit den Linien:** Kartons werden zu weißen Flächen und wachsen mit den Parkplatzlinien zusammen.
- **Boden und toter Winkel sind nicht mehr unterscheidbar:** Asphalt und der Bereich außerhalb des Kamerasichtfelds sind dann beide schwarz.
- **Ein fester Schwellwert passt nicht zu jedem Licht.** Ein CNN lernt solche Filter selbst und angepasst.

Sinnvoll ist die Binarisierung nur für den Transfer von Simulations- auf echte Daten. Für einen Vergleich in der Arbeit eignet sich eine 3-Kanal-Eingabe aus Grau-, Binär- und Kantenbild.

---

## 5. Vergleich der Modellkonzepte

### 5.1 Kandidaten

| Konzept | Ausgabe | Repräsentation des Platzes | Vortraining | Größe | Bewertung für diese Aufgabe |
|---|---|---|---|---|---|
| **Klassische Bildverarbeitung** (altes Paket `parking_spot`) | Linien/Ecken nach Schwellwert | Ecken aus Linienschnitt | – | – | schnell, erklärbar, aber empfindlich gegen Licht, Hindernisse auf Linien und teilweise Sichtbarkeit; keine Klassifikation frei/besetzt |
| **Markierungspunkt-Netze** (DMPR-PS, GCN-Parking-Slot) | Markierungspunkte mit Richtung, Paarbildung | T/L-Markierungen | ImageNet | mittel | für reale Parkplätze mit T/L-Markierungen entworfen; die Strecke hat geschlossene Rechtecke und X-Flächen; Paarbildung zusätzlich nötig |
| **VPS-Net** (Detektion + Belegt-Klassifikation) | Markierungspunkte + Klassifikator | Platz aus Punkten | ja | groß, zweistufig | kommt dem Problem fachlich am nächsten, aber zweistufig und für Rundumsicht-Bilder großer Fahrzeuge |
| **ParkingNet** (eigenes Netz, CenterNet-Stil) | Heatmap je Klasse + 8 Eckenversätze | Mittelpunkt + 4 Ecken | **keins** | 0,61 Mio. Parameter, Eingabe 400 × 320 | sehr klein und schnell; ohne Vortraining bei ~500 Bildern deutlich unterlegen; **Baseline** |
| **YOLO-OBB** (gedrehte Boxen) | Mittelpunkt, Breite, Länge, Winkel | Rechteck direkt | COCO/DOTA | n: ~3 Mio. | passt zur Rechteckform; keine Sichtbarkeit je Ecke, abgeschnittene Plätze lassen sich schlecht darstellen; **Vergleichskandidat** |
| **YOLO-Seg** (Instanzsegmentierung) | Maske je Platz | Polygon aus Maske | COCO | n: ~3 Mio., mehr Rechenaufwand | robuste Fläche, aber Ecken müssen nachträglich gefittet werden; Annotation als Polygon bereits vorhanden |
| **YOLO11n-Pose** | Box, Klasse, 4 Keypoints mit Sichtbarkeit | 4 Ecken | COCO-Keypoints | **2,66 Mio. Parameter, 6,8 GFLOPs** | bisher eingesetzt; Ecken und Sichtbarkeit direkt; NMS nötig |
| **YOLO26n-Pose** | wie YOLO11-Pose + Unsicherheit je Keypoint | 4 Ecken | COCO-Keypoints | **~3,7 Mio. Parameter, ~10,7 GFLOPs** (Standardkonfiguration mit 17 Keypoints) | **gewählt**, Begründung in 5.2 |

Parameterzahlen: ParkingNet und YOLO11n-Pose aus den eigenen trainierten Modellen, YOLO26n-Pose aus `cfg/models/26/yolo26-pose.yaml` der installierten Ultralytics-Version. Mit 4 statt 17 Keypoints und 6 statt 80 Klassen sind es etwas weniger.

### 5.2 Begründung für YOLO26n-Pose

1. **Passende Repräsentation:** Ein Parkplatz ist genau ein Objekt mit 4 Eckpunkten. Pose-Modelle liefern Klasse, Box und Keypoints in einem Durchlauf (Maji et al. 2022), ohne zweite Stufe und ohne Paarbildung wie bei Markierungspunkt-Netzen.
2. **Sichtbarkeit je Ecke:** Die Keypoint-Sichtbarkeit (`v ∈ {0, 1, 2}`) bildet abgeschnittene und geschätzte Ecken direkt ab. Unbekannte Ecken (`v=0`) gehen nicht in den Fehler ein, und das Netz gibt pro Ecke eine Konfidenz aus. Die Pipeline nutzt das, um unsichere Ecken auszublenden (Abschnitt 7.1). YOLO-OBB und Segmentierung bieten das nicht.
3. **Unsicherheit je Keypoint (RLE):** Der Pose-Kopf von YOLO26 (`Pose26`) sagt zusätzlich eine Streuung σ je Keypoint voraus und wird mit **Residual Log-Likelihood Estimation** trainiert, einer Lernverteilung über einen Normalizing Flow (RealNVP), nach Li et al. (ICCV 2021). Das verbessert die Lokalisierung bei unsicheren Punkten und liefert langfristig ein Maß dafür, wie sicher eine Ecke liegt. Ob und wie σ bei der Inferenz ausgelesen werden kann, ist noch zu prüfen.
4. **NMS-frei (end-to-end):** YOLO26 hat zusätzlich einen 1-zu-1-Kopf (`end2end: True`) und braucht keine Non-Maximum-Suppression.
   - Die Latenz ist deterministischer.
   - Es entsteht kein Schwellwert-Tuning für die NMS.
   - Das bisherige Risiko, dass derselbe Platz mit zwei Klassen erkannt wird (klassenweise NMS), sinkt.
5. **Ohne DFL (`reg_max: 1`):** Die Box-Regression ist einfacher, das vereinfacht den Export (ONNX/OpenVINO) für den NUC.
6. **Vortraining auf COCO-Keypoints:** Bei ~500–1500 Bildern ist das der entscheidende Vorteil gegenüber ParkingNet.
7. **Klein genug:** Das n-Modell (~10 GFLOPs) ist für CPU-Inferenz auf dem NUC realistisch, gemessen werden muss trotzdem (Abschnitt 13.4).

Gegenüber YOLO11n-Pose kostet YOLO26n etwa 40 % mehr Parameter und ~55 % mehr GFLOPs. Ob sich das in genaueren Ecken auszahlt, ist der zentrale Vergleich in der Evaluation (Abschnitt 12.3).

### 5.3 Warum Keypoints und nicht direkt Mittelpunkt und Winkel?

Die Lage des Platzes wird **nicht** direkt vom Netz übernommen, sondern aus den Ecken mit bekannter Geometrie gefittet (Abschnitt 7.2). Dadurch:
- werden einzelne falsche Ecken erkannt (Residuum) und können ausgeblendet werden,
- reichen 3 Ecken für einen abgeschnittenen Platz,
- ist die Form immer ein Rechteck mit Regelmaßen, das Netz liefert nur die Lage.

---

## 6. Detektor: YOLO26n-Pose

### 6.1 Architektur (Ultralytics YOLO26, Skalierung n)

```text
Eingang 800 × 717 (Graubild, 3-kanalig dupliziert), letterbox auf 800
  │
  ▼  Backbone
Conv 3×3/2 → Conv 3×3/2 → C3k2 → Conv/2 (P3/8) → C3k2 → Conv/2 (P4/16) → C3k2
  → Conv/2 (P5/32) → C3k2 → SPPF → C2PSA (Aufmerksamkeit auf P5)
  │
  ▼  Neck (PAN/FPN)
Upsample + Concat (P4) → C3k2 → Upsample + Concat (P3) → C3k2 [P3, klein]
  → Conv/2 + Concat → C3k2 [P4, mittel] → Conv/2 + Concat → C3k2 [P5, groß]
  │
  ▼  Kopf Pose26 auf P3, P4, P5
Box (reg_max = 1, ohne DFL) │ Klasse (6) │ Keypoints 4 × (x, y, Sichtbarkeit) │ σ je Keypoint (RLE)
  1-zu-viele-Kopf (Training) und 1-zu-1-Kopf (end-to-end, NMS-frei)
```

Bei 800 px Eingang hat P3 eine Rasterweite von 8 px ≈ 24 mm. Ein 90°-Platz (350 × 500 mm ≈ 117 × 167 px) liegt damit gut im Bereich der Merkmalskarten P3 bis P4.

### 6.2 Ausgabe und Weiterverarbeitung

| Ausgabe | Verwendung |
|---|---|
| Klasse (6 Labels) + Score | Art (90°/parallel) und Status (frei/besetzt/gesperrt) über `classes` in `planning.yaml`; Score-Schwelle 0,3 |
| 4 Keypoints (x, y) | Ecken in px → mm über die Ansicht |
| Keypoint-Konfidenz | Sichtbarkeit der Ecke (`corner_visibility: keypoint_confidence`, Schwelle 0,5) |
| Box | nicht verwendet |

### 6.3 Klassen

| ID | Label | Art | Status |
|---|---|---|---|
| 0 | `Parking_90_free` | perpendicular | free |
| 1 | `Parking_90_besetzt` | perpendicular | occupied |
| 2 | `Parking_90_gesperrt` | perpendicular | blocked |
| 3 | `Parking_Parallel_Frei` | parallel | free |
| 4 | `Parking_Parallel_Besetzt` | parallel | occupied |
| 5 | `Parking_Parallel_Gesperrt` | parallel | blocked |

Art und Status stecken bewusst in einer gemeinsamen Klasse. So entsteht ein einziger Kopf, und die Seitenregel (90° links, parallel rechts) lernt das Netz mit. Horizontales Spiegeln ist deshalb in der Augmentierung verboten.

### 6.4 Umstellung von YOLO11n auf YOLO26n

In `notebooks/train_yolo_pose.ipynb`, Zelle „3. Training“:

```python
BASE_MODEL = "yolo26n-pose.pt"   # bisher "yolo11n-pose.pt"
```

Der übrige Code bleibt gleich:
- **Export, Training, Speichern** unverändert. `optimizer="auto"` wählt bei weniger als 10 000 Iterationen AdamW, darüber den neuen MuSGD-Optimierer. Bei unserer Datenmenge ist es also AdamW.
- **Inferenz in der Node** über `ultralytics.YOLO`, ebenfalls unverändert.

Damit beide Modelle verglichen werden können, wird das YOLO26-Modell unter einem eigenen Namen gespeichert, z. B. `models/yolo26_pose_parking_bev.pt`. Dann trägt man es in `config/model.yaml` unter `yolo_pose.parking_bev.model_path` ein.

---

## 7. Nachverarbeitung: Geometrie und Tracking

### 7.1 Sichtbarkeit der Ecken

| Modus (`corner_visibility`) | Regel | Einsatz |
|---|---|---|
| `image_area` | Ecke gilt als verdeckt, wenn sie im schwarzen Bereich der BEV oder weniger als 8 px davon entfernt liegt | Modelle ohne gelernte Sichtbarkeit (altes `bev`-Modell, ParkingNet) |
| `keypoint_confidence` | Ecke gilt als verdeckt, wenn die Keypoint-Konfidenz des Netzes < 0,5 ist | Modelle, die mit der Sichtbarkeit aus `yolo_export` trainiert sind (`parking_bev`) |

Begründung für `keypoint_confidence`: Beim neuen Modell liegen Ecken auf dem Bildrand (`v=0`) im Median 115 px daneben, haben aber eine niedrige Konfidenz (Median 0,45, zu 73 % unter 0,5). Geschätzte Ecken im schwarzen Bereich (`v=1`) sind dagegen gut gelernt (Median 27 px) und dürfen nicht pauschal verworfen werden.

### 7.2 Rechteck-Fit mit bekannter Geometrie

Gesucht ist die Pose (x, y, ψ) eines Rechtecks mit Nennmaß, das die erkannten Ecken bestmöglich trifft.

1. **Zuordnung:** Die Ecken sind nach Bildposition sortiert, und die Transformation Bild → Fahrzeug spiegelt die Umlaufrichtung. Deshalb werden alle **8 Zuordnungen** (4 Startecken × 2 Richtungen) geprüft, und die mit dem kleinsten Residuum gewinnt.
2. **Gewichteter Kabsch (Umeyama) in 2D:** Die Translation bildet die gewichteten Schwerpunkte aufeinander ab. Der Drehwinkel folgt geschlossen aus
   `ψ = atan2( Σ wᵢ (aᵢ × bᵢ), Σ wᵢ (aᵢ · bᵢ) )`
   mit den zentrierten Schablonenpunkten aᵢ und Detektionspunkten bᵢ. Verdeckte Ecken haben das Gewicht 0,01. Das Residuum zählt nur über sichtbare Ecken.
3. **Größe:**
   - **4 Ecken sichtbar:** zweimal neu fitten mit den gemessenen Maßen. Das Residuum misst dann die Rechteckigkeit.
   - **3 Ecken sichtbar:** Länge und Breite aus den beiden sichtbaren Kanten an der Ecke gegenüber der verdeckten. Das ist für Parallelplätze mit variabler Länge nötig.
4. **Plausibilitätsprüfung gegen das Regelwerk:**

| Prüfung | 90° | parallel |
|---|---|---|
| sichtbare Ecken | ≥ 3 | ≥ 3 |
| Residuum (Rechteckigkeit) | ≤ 50 mm | ≤ 50 mm |
| Länge | 500 mm ± 35 % | nur ≥ 400 mm (variabel) |
| Breite | 350 mm ± 35 % | 300 mm ± 35 % |

Nicht bestandene Detektionen zählen für die Klasse, aber nicht für die Lage. Ihr Grund erscheint im Debug-Bild: `cut off`, `not rectangular`, `wrong size`.

### 7.3 Tracking

| Baustein | Umsetzung |
|---|---|
| Bezugssystem | global (mit `/pose_estimation/pose`), sonst Fahrzeug-KS (Rosbags ohne Pose); beim Wechsel Reset |
| Prädiktion | alle Plätze statisch → gemeinsame Verschiebung je Frame, geschätzt als Median der Zuordnungsresiduen (konstante Geschwindigkeit, wie der Prädiktionsschritt eines Kalman-Filters); klingt ohne Zuordnung ab |
| Zuordnung | greedy nach kleinstem Mittelpunktabstand, gleiche Art, Gating 250 mm |
| Glättung | gleitender Mittelwert (α = 0,4) für Mittelpunkt, Länge, Breite; Winkel als doppelter Winkel (Rechteck um 180° symmetrisch) |
| Klasse | Mehrheitsentscheid über die letzten 15 Detektionen |
| Bestätigung | ≥ 3 Detektionen, davon ≥ 2 geometrisch zuverlässig |
| Löschen | nach 10 Frames ohne Detektion |

Eine Prädiktion mit Rotation (Kabsch über die Platzmittelpunkte) wurde getestet und wegen zu großen Rauschens verworfen. Sie erzeugte 5× so viele Tracks.

---

## 8. Zielauswahl und Einparkplanung

### 8.1 Prüfungen je Track

1. **Bestätigt?** (Abschnitt 7.3)
2. **Frei?** (Mehrheitsentscheid)
3. **Zielpose:** Das Fahrzeug wird im Platz zentriert. Bezugspunkt ist die Hinterachsmitte (Einspurmodell):
   `p_HA = c_Platz − d · (cos ψ, sin ψ)`, mit `d = L/2 − l_h = 115 mm`
   - 90° vorwärts: ψ zeigt vom Fahrzeug weg in den Platz. 90° rückwärts: ψ zeigt zur Straße.
   - parallel: ψ liegt in Fahrtrichtung.
4. **Footprint-Test:** Das Fahrzeugrechteck plus 20 mm Sicherheitsabstand je Seite muss in der Zielpose **vollständig im gemessenen Platz** liegen. Geprüft wird der kleinste vorzeichenbehaftete Abstand zu den Platzkanten. Das entspricht der Regel „gesperrt, wenn das Fahrzeug nicht vollständig hineinpasst“ und ist strenger als „alle Räder im Platz“.
5. **Mindestlänge parallel** (Einzug-Manöver, Paromtchik & Laugier 1996):
   `L_min = l_h + √( (R + b/2)² + l_v² − (R − b/2)² )`
   Mit R = 371, b = 200, l_v = 300, l_h = 70 mm ergibt sich **L_min = 558 mm**. Kürzere Plätze (400–558 mm) sind laut Regelwerk gültig, aber nicht in einem Zug erreichbar. Da die Plätze in Fahrtrichtung länger werden, folgt ein passender.
6. **Bahn existiert** (8.2).

### 8.2 Planer

Beide Planer verwenden nur Geraden und Kreisbögen mit R = R_min. Jeder Abschnitt ist mit konstantem Lenkwinkel fahrbar: δ = atan(Radstand · κ).

**90°-Plätze: Gerade – Bogen – Gerade** (Dubins-Fall SCS, Dubins 1957)

Der Bogenmittelpunkt hat zu beiden Geraden, der aktuellen Fahrtrichtung und der Platzachse, den Abstand R:

`P₀ + t·h₀ + s·R·n₀ = G − u·h_G + s·R·n_G`

Das ist ein lineares 2×2-System für die Anfahrtslänge t und die Einfahrtslänge u.
- u < 0: Der Platz liegt seitlich zu nah.
- t < 0: Der Wendepunkt ist überfahren, die Bahn beginnt mit Zurücksetzen (Umschaltpunkt).
- Erst vorwärts, dann rückwärts versuchen.

**Parallelplätze: Rückwärts-S-Kurve** (Paromtchik & Laugier 1996; Vorobieva et al. 2015)

Konstruktion rückwärts vom Ziel:
- Der letzte Bogen liegt auf dem Kreis um C₁ = (0, s·R).
- Der erste Kreis berührt ihn: C₂ = C₁ + 2R·(sin α, −s·cos α).
- C₂ muss den Abstand R zur Anfahrtsgeraden haben. Daraus folgt geschlossen `α = s·φ + arccos(−s·k)`, und der zweite Winkel ist β = α − s·φ.

Ein Winkelfehler φ bis 15° wird damit exakt ausgeglichen. Für φ = 0 ergibt sich der Lehrbuchfall `α = β = arccos(1 − Δy / 2R)`. Steht das Fahrzeug schon in Linie mit dem Platz (Querversatz < 30 mm), wird nur gerade gefahren.

### 8.3 Auswahl

Gewählt wird der machbare Kandidat mit der kürzesten Bahn, optional eingeschränkt auf eine Art. Dazu kommt **Hysterese**: Ein Ziel bleibt, solange es machbar ist. Das hat die Zielwechsel auf den Test-Bags halbiert. Bahn und Ziel werden in jedem Frame neu geplant.

### 8.4 Bewusst nicht umgesetzt

| Verfahren | Grund |
|---|---|
| vollständiges Reeds-Shepp (Reeds & Shepp 1990) | für zwei bekannte Manöver nicht nötig; Kandidat für mehrzügige Manöver |
| Hybrid A\* (Dolgov et al. 2010) | keine freien Hindernisse in der Parkzone |
| Klothoiden | stetige Krümmung wäre regelungsfreundlicher; nächster Ausbauschritt |
| Kollisionsprüfung entlang der Bahn | bisher nur Footprint im Ziel |
| Regelung | nicht Teil des Prototyps |

---

## 9. Schnittstellen

### 9.1 Topics der `parking_perception_node`

| Richtung | Topic | Typ | Inhalt |
|---|---|---|---|
| ein | `/camera/image/bev` oder `/camera/image/undistorted` | `sensor_msgs/Image` | je nach Parameter `view` |
| ein | `/pose_estimation/pose` | `geometry_msgs/Pose` | Fahrzeugpose (mm, `orientation.z` = Gierwinkel) |
| aus | `/parking/path` | `nav_msgs/Path` | Posen der Hinterachse alle 20 mm, Fahrzeug-KS zum Bildzeitpunkt, mm, Quaternion; **leer ohne Ziel** |
| aus | `/parking/target_pose` | `geometry_msgs/PoseStamped` | Zielpose der Hinterachse |
| aus | `/parking/path_segments` | `std_msgs/Float32MultiArray` | **κ(s)**: pro Segment `[Länge mm, Krümmung 1/mm, Richtung ±1]` |
| aus | `/parking/debug_image` | `sensor_msgs/Image` | Debug-Bild (nur `debug:=True`) |

### 9.2 Bahn als κ(s)

Die Bahn wird als stückweise konstante Krümmung mit expliziter Fahrtrichtung übergeben, nicht als Funktion y = f(x). Eine Funktion kann Rückwärtszüge und Umschaltpunkte nicht abbilden. Das Format entspricht der Darstellung von Dubins- und Reeds-Shepp-Pfaden. Ein Pure-Pursuit- (Coulter 1992) oder Stanley-Regler (Thrun et al. 2006) kann alternativ `/parking/path` folgen.

### 9.3 Parameter (Auszug)

| Parameter | Werte |
|---|---|
| `view` | `bev`, `undistorted`, `parking_bev` |
| `image_subscriber` | muss zu `view` passen |
| `use_pose` | Tracking im globalen KS, wenn eine Pose empfangen wird |
| `frame_id` | Frame der Bahn, Standard `vehicle` |

---

## 10. Daten- und Annotationskonzept

### 10.1 Werkzeugkette

```text
Rosbag ──extract_frames.py --view parking_bev --every 5──► PNG (800×717)
   ──CVAT (Polygon, 4 Punkte, Label = Klasse)──► Export "CVAT for images 1.1"
   ──data/BEV_PARKING/<Task>/annotations.xml + Bilder──► Notebook ──yolo_export──► YOLO-Format
```

Werkzeuge im Paket:
- `extract_frames.py`: Frames in der gewählten Ansicht.
- `convert_annotations_to_parking_bev.py`: alte Annotationen umrechnen.
- `cvat_function_yolo.py`: Vorannotation mit `image_size`.
- `run_pipeline_on_bag.py`: Offline-Bewertung.

Zusammenarbeit über den gemeinsamen CVAT-Server, erreichbar über Tailscale.

### 10.2 Annotationsregeln

1. Ein Polygon mit **genau 4 Punkten** pro Platz, die Reihenfolge ist egal. Das Label ist die Klasse.
2. Annotieren, sobald **im Bild selbst** sicher erkennbar ist, dass es ein Platz ist und welche Klasse er hat. Faustregel: mindestens zwei Ecken bzw. eine komplette Kante.
3. **Immer das vollständige Rechteck an der realen Position:**
   - Ecken im schwarzen Bereich an die verlängerten Linien setzen. Orientierung: 90° ≈ 117 × 167 px, parallel 100 px tief.
   - Ecken außerhalb des Bildes auf den Bildrand setzen.
4. Ecke = Schnittpunkt der **Linienmitten**.
5. Jedes Bild vollständig annotieren. Ein vergessener Platz zählt als „kein Parkplatz“.

### 10.3 Sichtbarkeit im Export (`parking/training/yolo_export.py`)

| v | Bedeutung | Bestimmung | Training |
|---|---|---|---|
| 2 | sichtbar | Ecke im Kamerabereich des Bildes | Ort und Sichtbarkeit gelernt |
| 1 | geschätzt | Ecke im schwarzen Bereich | Ort gelernt |
| 0 | unbekannt | Ecke ≤ 2 px am Bildrand | Ort **nicht** gelernt |

Datensatz BEV v1 (Bag 13_32_18): 983 / 126 / 507 Ecken. Die bisherigen Datensätze v4–v6 in der alten BEV hatten bei **92 %** der Plätze abgeschnittene Ecken, weil der sichtbare Ausschnitt annotiert war. Das war die Hauptursache schlechter Geometrie im alten Modell.

### 10.4 Aufteilung

- **Training und Validierung:** Blöcke aus 10 aufeinanderfolgenden Bildern je Bag, 20 % der Blöcke gehen in die Validierung.
- **Testset:** ein vollständig zurückgehaltener Bag (13_38_33). Aufeinanderfolgende Frames sind stark korreliert, die Validierung ist deshalb optimistisch.

---

## 11. Trainingskonzept

### 11.1 Hyperparameter (Notebook `train_yolo_pose.ipynb`)

| Parameter | Wert | Begründung |
|---|---|---|
| `BASE_MODEL` | `yolo26n-pose.pt` (bisher `yolo11n-pose.pt`) | Abschnitt 5.2 |
| `imgsz` | 800 (`parking_bev`), 640 (`bev`) | volle Breite, keine Verkleinerung |
| `epochs` / `patience` | 100–150 / 30 | 30 Epochen waren nachweislich zu wenig |
| `batch` | 8 | Speicher Laptop/Mac |
| `optimizer` | auto (bei unserer Datenmenge AdamW) | Ultralytics-Logik |
| `fliplr` | 0,0 | Seitenregel 90° links / parallel rechts |
| `hsv_h`, `hsv_s` | 0,0 | Graubilder |
| `hsv_v` | 0,4 (Option 0,5) | Lichtwechsel |
| `degrees` | 15 | gedrehte Plätze beim Einparken |
| `scale` | 0,2 | fester Maßstab der Vogelperspektive |
| `mosaic` / `close_mosaic` | 1,0 / 10 | Standard, in den letzten Epochen ohne Mosaic |

### 11.2 Hardware

| Rechner | Zeit pro Epoche |
|---|---|
| Laptop-CPU | 230 s (94 Bilder, 800 px) |
| Mac M5 (MPS) | ~26 s (438 Bilder, 640 px, bisheriges Training) |

Training auf dem Mac, Inferenz-Messung auf dem NUC.

### 11.3 Iterativer Datenkreislauf

```text
annotieren (wenige Bags) → trainieren → Vorannotation der nächsten Bags (cvat_function_yolo)
   → korrigieren → neu trainieren → auf dem Testset bewerten → …
```

---

## 12. Evaluationskonzept

### 12.1 Metriken ohne Ground Truth (sofort verfügbar, `run_pipeline_on_bag.py`)

| Metrik | Aussage |
|---|---|
| Anteil zuverlässiger Detektionen (`ok`) je Art | Geometriequalität des Netzes |
| Gründe für Verwerfen (`cut off`, `not rectangular`, `wrong size`) | Fehlerart |
| Frames mit Ziel je Art | Nutzbarkeit für das Einparken |
| Klassenabweichung vom Mehrheitsentscheid | Stabilität der Klassifikation |
| Doppel-Tracks, Zielwechsel | Tracking- und Planungsstabilität |
| Laufzeit Detektion / Pipeline | Echtzeitfähigkeit |

### 12.2 Metriken mit Ground Truth (Testset annotieren)

| Metrik | Einheit |
|---|---|
| Eckenfehler je Sichtbarkeitsstufe | mm |
| Mittelpunktfehler, Winkelfehler nach Rechteck-Fit | mm, ° |
| Precision, Recall, F1 je Klasse | – |
| Rate korrekter Ziele (frei, passend, richtige Seite) | % |
| Latenz auf dem NUC (Ansicht, Netz, Pipeline) | ms |

`mAP` allein ist nicht aussagekräftig. Die Pose-mAP ignoriert unbekannte Ecken und meldete beim untertrainierten Modell 0,995, während dieselben Ecken im Median 115 px daneben lagen.

### 12.3 Geplante Vergleiche für die Arbeit

| Vergleich | Variablen |
|---|---|
| Modelle | ParkingNet (Baseline) vs. YOLO11n-Pose vs. **YOLO26n-Pose** vs. YOLO26n-OBB |
| Ansicht | `bev` vs. `parking_bev` (vs. `undistorted`) |
| Eingabe | Grau vs. binär vs. 3-Kanal (Grau, Binär, Kante) |
| Sichtbarkeit | `image_area` vs. `keypoint_confidence` |
| Nachverarbeitung | Netz-Ecken roh vs. Rechteck-Fit; mit/ohne Tracking |

---

## 13. Aktueller Stand und Messergebnisse

### 13.1 Umgesetzt und getestet

- Pipeline, Planer, ROS-Anbindung, Debug-Bild, Offline-Tool.
- 37 Unit-Tests, ruff (Workspace-Konfiguration) ohne Befund.
- Live-Test mit ROS und Rosbag: alle Topics veröffentlicht, κ(s) plausibel (z. B. 675 mm gerade, 580 mm Bogen mit R = 371 mm, 148 mm gerade).
- Planer: Die integrierte Bahn endet in den Tests exakt in der Zielpose, auch mit Winkelfehler und Umschaltpunkt.

### 13.2 Altes Modell (YOLO11n-Pose, Ansicht `bev`, Datensätze v4–v6, 30 Epochen)

| Kennzahl | Wert |
|---|---|
| Validierung Pose-mAP50 / mAP50-95 | 0,966 / 0,847 |
| zuverlässige Detektionen auf ungesehenen Bags | 17–23 % |
| Klassenabweichung vom Mehrheitsentscheid | 0–3 % |
| Frames mit Ziel (Regelmaße) auf 13_37_32 / 13_35_42 / 13_38_33 | 159 / 241 / 91 |
| Laufzeit Detektion / Pipeline (Laptop-CPU) | ~60 ms / 3–5 ms |

Die Klassifikation ist stabil, die Geometrie war das Problem, Ursache waren die abgeschnittenen Annotationen.

### 13.3 Erstes Modell in `parking_bev` (YOLO11n-Pose, 1 Bag, 30 Epochen CPU)

| Kennzahl | Wert |
|---|---|
| Validierung Pose-mAP50 / mAP50-95 | 0,995 / 0,69 (mAP-Kurven am Ende noch steigend) |
| Eckenfehler auf dem Trainings-Bag: sichtbar / geschätzt / Bildrand | 36 / 27 / 115 px (Median) |
| Testset 13_38_33: zuverlässig parallel / 90° | 65 / 0 |
| Testset: Frames mit Ziel (`keypoint_confidence`) | 19 (parallel), vorher 0 |

Bewertung: untertrainiert, ein Bag, keine Beispiele für „90° gesperrt“. Die 90°-Plätze scheitern an den immer unbekannten äußeren Ecken (Abschnitt 14).

### 13.4 Laufzeit

| Konfiguration | Detektion (Laptop-CPU) |
|---|---|
| YOLO11n, `bev`, 640 | ~60 ms |
| YOLO11n, `parking_bev`, 800 | ~95–100 ms |
| YOLO11n, `parking_bev`, 864 | ~200 ms |
| YOLO26n | noch zu messen (rechnerisch ~1,5× YOLO11n) |

Für N1 (≥ 10 Hz) ist auf dem NUC voraussichtlich ein Export nach ONNX/OpenVINO oder `imgsz=640` nötig (Abschnitt 13.5).

### 13.5 Rechenbudget und Deployment auf dem NUC

**Messung: Wo geht die Zeit hin?** Gemessen am 09.11.2026 auf einem Laptop (Intel i5-1145G7, 4 Kerne, PyTorch-CPU), Bag 13_35_42 in der Parking-BEV, 239 Frames, 2,4 Detektionen pro Frame:

| Schritt | Zeit pro Bild |
|---|---|
| Parking-BEV berechnen (`view.prepare`, 1 Thread) | 4,8 ms |
| Netz YOLO11n-Pose, 800 px, 1 / 2 / 4 Threads | 164 / 111 / 83 ms |
| Netz YOLO11n-Pose, 640 px, 1 / 2 / 4 Threads | 95 / 67 / 52 ms |
| Python-Pipeline gesamt (Fit, Tracking, Footprint, Planung) | **2,4 ms** Mittel, 5,9 ms p95, 15 ms max |
| davon Rechteck-Fit | ~80 % der Pipeline |
| davon Footprint-Test und Planung | ~0,04 ms je Platz |

**Ergebnis:** Die geometrische Nachverarbeitung in Python ist mit ~2–3 % der Rechenzeit **kein Engpass**. Über 95 % der Zeit verbraucht das Netz. YOLO26n hat ~55 % mehr GFLOPs als YOLO11n und wird ohne weitere Maßnahmen entsprechend langsamer sein.

**Zielbudget für 10 Hz (100 ms pro Bild)** bei geteilter CPU (camera_preprocessing, Spur- und Objekterkennung laufen parallel):

| Schritt | Budget |
|---|---|
| Parking-BEV | ≤ 5 ms |
| Netz | ≤ 60 ms |
| Python-Pipeline | ≤ 10 ms |
| Veröffentlichen, Debug aus | ≤ 2 ms |
| Reserve | ~20 ms |

**Maßnahmen, nach Wirkung sortiert:**

1. **Netz mit OpenVINO statt PyTorch ausführen.** OpenVINO ist Intels Inferenz-Laufzeit für genau diese CPUs (und die integrierte GPU). Ultralytics exportiert direkt:
   ```python
   from ultralytics import YOLO
   YOLO("models/yolo_pose_parking_bev.pt").export(format="openvino", imgsz=640)
   # optional INT8 (Kalibrierung mit den Trainingsbildern):
   YOLO("models/yolo_pose_parking_bev.pt").export(
       format="openvino", imgsz=640, int8=True, data="data/yolo/<datensatz>/data.yaml"
   )
   ```
   - Das Ergebnis ist ein Ordner `..._openvino_model/`. `ultralytics.YOLO` lädt ihn wie eine `.pt`-Datei. Dafür in `config/model.yaml` `model_path` auf den Ordner setzen, der Code der Node bleibt gleich.
   - Voraussetzung: `pip install openvino` (ohne `-U`, wegen numpy und cv_bridge).
   - Prüfen, ob `setup.py` den Ordner unter `models/` mitinstalliert.
   - Den Geschwindigkeitsgewinn und bei INT8 den Genauigkeitsverlust (Eckenfehler, Abschnitt 12.2) auf dem Test-Bag messen.
   - YOLO26 ist NMS-frei, der exportierte Graph enthält also keine NMS-Nachbearbeitung.
2. **Eingabe 640 statt 800 px** (`imgsz` im Training und `image_size` in `config/model.yaml`). Das spart gemessen ~40 %. Ein 90°-Platz ist dann immer noch ~93 × 133 px groß.
3. **Nur rechnen, wenn es nötig ist.**
   - Die Node über den vorhandenen Zustand von `SmartyNode` (`state`: ACTIVE/INACTIVE) nur in der Parkzone aktivieren, ausgelöst durch die Zustandsmaschine nach dem Parkschild.
   - Während der Suche reichen 5–10 Hz: Das Tracking mit Pose überbrückt die Lücken. Umsetzbar, indem `image_callback` nur jedes n-te Bild verarbeitet oder einen Mindestabstand in der Zeit einhält.
   - Während des Manövers die Detektion drosseln. Das Ziel ist dann getrackt und die Bahn im globalen KS fixiert (Abschnitt 14, Punkt 7).
4. **Rechenleistung begrenzen und Leerlauf vermeiden.**
   - `torch.set_num_threads(2)` bzw. `ov_config={"INFERENCE_NUM_THREADS": 2}` bei OpenVINO, damit andere Nodes genug CPU behalten.
   - Auf dem Fahrzeug `debug:=False`: Das Zeichnen und Senden des Debug-Bilds kostet zusätzlich ~3 ms plus Übertragung.
5. **Python-Pipeline nur bei Bedarf optimieren.** Sie liegt weit im Budget. Sollte sie bei vielen Plätzen wachsen, ist der Rechteck-Fit der einzige Hebel: die 8 Zuordnungen als ein numpy-Block rechnen statt in einer Schleife, und die Verfeinerung nur für Detektionen, die die Grobprüfung bestehen.
6. **Option Coral-TPU:** Die vorhandene Edge-TPU bräuchte einen INT8-TFLite-Export und wird bereits von `object_detection` belegt. Deshalb nur, wenn OpenVINO nicht reicht.

**Messen auf dem NUC:** Die Node loggt pro Bild die Zeiten `view`, `prediction`, `planning`, `publish` und `debug` (Timer aus `smarty_utils`). Diese Werte im laufenden Gesamtstack aufnehmen, einmal mit PyTorch und einmal mit OpenVINO, jeweils 640 und 800 px. Erst dann `imgsz` und das Modell (YOLO11n oder YOLO26n) endgültig festlegen.

---

## 14. Grenzen, Risiken und offene Punkte

| # | Thema | Auswirkung | Maßnahme |
|---|---|---|---|
| 1 | Äußere Ecken der 90°-Plätze liegen außerhalb der Ansicht (y > 1,2 m) | nie gelernt (`v=0`); mit „≥ 3 sichtbare Ecken“ werden 90°-Plätze nie zuverlässig | Fit mit 2 Ecken an der Fahrbahn zulassen: bei Breite 350 mm, Tiefe 500 mm und bekannter Seite ist das Rechteck eindeutig bestimmt |
| 2 | Parallelplätze messen ~412 mm statt 300 mm Tiefe | Verwerfen als „wrong size“ (37 % > 35 % Toleranz) | Teststrecke nachmessen; ggf. eigene Planungs-Config für die Teststrecke |
| 3 | Rosbags ohne Pose | Tracking nur im Fahrzeug-KS getestet | Bags mit `/pose_estimation/pose` und Sensor-Topics aufnehmen |
| 4 | Fahrzeugmaße und Lage der Hinterachse aus Simulation bzw. altem Paket | Zielpose, Footprint, L_min | am Fahrzeug messen |
| 5 | wenig Daten, fehlende Klasse „90° gesperrt“ | schwache Generalisierung | weitere Bags annotieren, gezielt seltene Klassen |
| 6 | Laufzeit mit 800 px auf CPU | evtl. < 10 Hz | NUC messen, Export, kleinere Eingabe |
| 7 | Bahn bei jedem Frame im Fahrzeug-KS neu geplant | für die Ausführung muss die Bahn im globalen KS fixiert werden | mit Regelung umsetzen |
| 8 | Krümmungssprünge zwischen Gerade und Bogen | Regelungsfehler beim Übergang | Klothoiden |
| 9 | keine Kollisionsprüfung entlang der Bahn | Hindernisse 20–100 mm neben den 90°-Plätzen | Kollisionsprüfung mit besetzten/gesperrten Tracks |
| 10 | pyright (Teil von `make test`) konnte in der Entwicklungsumgebung nicht laufen | Typfehler möglich | lokal ausführen |

---

## 15. Roadmap

| Phase | Inhalt | Ergebnis |
|---|---|---|
| 1 | Restliche Bags in `parking_bev` annotieren (Vorannotation), Testset annotieren | Datensatz mit ≥ 4 Trainings-Bags + Ground Truth |
| 2 | Training YOLO11n-Pose und **YOLO26n-Pose**, je 100–150 Epochen auf dem Mac | Vergleich nach Abschnitt 12 |
| 3 | 2-Ecken-Fit für 90°-Plätze, Maße messen, Bags mit Pose aufnehmen | 90°-Ziele, globales Tracking |
| 4 | NUC: Export (ONNX/OpenVINO), Latenz messen, `imgsz` festlegen | Echtzeitnachweis |
| 5 | Regelung (Pure Pursuit/Stanley auf κ(s)), Bahn global fixieren, Kollisionsprüfung | Einparken auf dem Fahrzeug |
| 6 | Optional: YOLO26n-OBB, 3-Kanal-Eingabe, Klothoiden, Reeds-Shepp | wissenschaftliche Vergleiche |

---

## 16. Literatur

- Arun, Huang, Blostein: *Least-Squares Fitting of Two 3-D Point Sets*. IEEE TPAMI, 1987.
- Coulter: *Implementation of the Pure Pursuit Path Tracking Algorithm*. CMU-RI-TR-92-01, 1992.
- Dinh, Sohl-Dickstein, Bengio: *Density Estimation Using Real NVP*. ICLR, 2017.
- Do, Choi: *Context-Based Parking Slot Detection With a Realistic Dataset*. IEEE Access, 2020.
- Dolgov, Thrun, Montemerlo, Diebel: *Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments*. IJRR, 2010.
- Dubins: *On Curves of Minimal Length with a Constraint on Average Curvature*. American Journal of Mathematics, 1957.
- Huang et al.: *DMPR-PS: A Novel Approach for Parking-Slot Detection Using Directional Marking-Point Regression*. ICME, 2019.
- Kabsch: *A Solution for the Best Rotation to Relate Two Sets of Vectors*. Acta Crystallographica, 1976.
- Li, Bian, Liu, Zhang, Xu, Lu: *Human Pose Regression with Residual Log-likelihood Estimation*. ICCV, 2021 (arXiv:2107.11291).
- Li et al.: *Vacant Parking Slot Detection in the Around View Image Based on Deep Learning* (VPS-Net). Sensors, 2020.
- Maji et al.: *YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss*. CVPR Workshops, 2022.
- Min et al.: *Attentional Graph Neural Network for Parking-Slot Detection*. IEEE RA-L / ICRA, 2021.
- Paromtchik, Laugier: *Motion Generation and Control for Parking an Autonomous Vehicle*. IEEE ICRA, 1996.
- Rajamani: *Vehicle Dynamics and Control*. Springer, 2012.
- Reeds, Shepp: *Optimal Paths for a Car That Goes Both Forwards and Backwards*. Pacific Journal of Mathematics, 1990.
- Thrun et al.: *Stanley: The Robot that Won the DARPA Grand Challenge*. Journal of Field Robotics, 2006.
- Ultralytics: *YOLO26* und *Pose Estimation*, docs.ultralytics.com (Implementierung `ultralytics` 8.4.193: `Pose26`, `PoseLoss26`, `RLELoss`, `MuSGD`).
- Umeyama: *Least-Squares Estimation of Transformation Parameters Between Two Point Patterns*. IEEE TPAMI, 1991.
- Vorobieva, Glaser, Minoiu-Enache, Mammar: *Automatic Parallel Parking in Tiny Spots: Path Planning and Control*. IEEE T-ITS, 2015.
- Wang et al.: *YOLOv10: Real-Time End-to-End Object Detection* (NMS-freies Training mit 1-zu-1-Zuordnung). NeurIPS, 2024.
- Zhang et al.: *Vision-Based Parking-Slot Detection: A DCNN-Based Approach and a Large-Scale Benchmark Dataset* (DeepPS). IEEE TIP, 2018.
- Zhou, Wang, Krähenbühl: *Objects as Points* (CenterNet). arXiv, 2019.
- CAuDri-Challenge: *Regulations 2026* (`CAuDri-Challenge_Regulations_2026.pdf`).

---

## 17. Anhang

### 17.1 Dateien

| Datei | Inhalt |
|---|---|
| `parking/parking_perception_node.py` | ROS-Node |
| `parking/pipeline.py` | Orchestrierung ohne ROS |
| `parking/perception/view.py` | Ansichten `bev`, `undistorted`, `parking_bev` |
| `parking/perception/bev.py` | Parking-BEV (Homographie, px ↔ mm) |
| `parking/perception/model.py`, `yolo_model.py` | Detektoren, Sichtbarkeit je Ecke |
| `parking/perception/network.py` | ParkingNet (Baseline) |
| `parking/perception/geometry.py` | Pose2D, Kabsch, Rechteck-Fit |
| `parking/perception/tracker.py` | Tracking |
| `parking/perception/visualization.py` | Debug-Bild |
| `parking/planning/vehicle.py`, `path.py`, `planner.py`, `target.py` | Fahrzeug, Bahn κ(s), Planer, Prüfungen und Auswahl |
| `parking/training/dataset.py`, `yolo_export.py` | CVAT-Import, YOLO-Export mit Sichtbarkeit |
| `notebooks/train_yolo_pose.ipynb`, `train_parking_net.ipynb` | Training |
| `tools/extract_frames.py` | Frames aus Rosbags (`--view parking_bev`) |
| `tools/convert_annotations_to_parking_bev.py` | Annotationen in die Parking-BEV umrechnen |
| `tools/cvat_function_yolo.py` | Vorannotation in CVAT |
| `tools/run_pipeline_on_bag.py` | Offline-Test mit Statistik und Video |
| `test/test_planning.py` | Unit-Tests |

### 17.2 Konfiguration

| Datei | Inhalt |
|---|---|
| `config/ros_params.yaml` | Topics, `view`, `use_pose`, `frame_id`, Pfade |
| `config/model.yaml` | Detektor, Modell und `image_size` je Ansicht, `corner_visibility`, Score-Schwelle, Parking-BEV-Ausschnitt |
| `config/planning.yaml` | Klassen, Regelmaße, Rechteck-Fit, Tracking, Fahrzeug, Planer |

### 17.3 Befehle

```bash
cd ~/smarty_workspace && source install/setup.zsh
colcon build --symlink-install --packages-select parking

# Unit-Tests
cd src/parking && python3 -m pytest test/test_planning.py -q && cd ../..

# Frames in der Parking-BEV
python3 src/parking/tools/extract_frames.py <bag> src/parking/data/frames_parking_bev/<bag> --every 5 --view parking_bev

# Offline-Bewertung auf dem Testset mit Video
python3 src/parking/tools/run_pipeline_on_bag.py ~/rosbags/rosbag2_2026_10_28-13_38_33 /tmp/parking_eval \
    --view parking_bev --video --show-detections

# Live (view: "parking_bev", image_subscriber: "/camera/image/undistorted" in ros_params.yaml)
ros2 launch camera_preprocessing camera_preprocessing.launch.py
ros2 launch parking parking_perception.launch.py debug:=True
ros2 bag play ~/rosbags/rosbag2_2026_10_28-13_38_33
```
