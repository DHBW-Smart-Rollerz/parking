# Parkplatzerkennung: kompletter Workflow

Diese Anleitung fasst den gesamten Ablauf zusammen: Rosbag ansehen, Kamera kalibrieren, Bilder extrahieren, in CVAT annotieren, Modelle trainieren und als ROS-Node auf einem Rosbag testen.

Alle Befehle gehen davon aus, dass der Workspace unter `~/smarty_workspace` liegt. Wo ROS gebraucht wird, sind die Befehle vorher gesourct:

```bash
cd ~/smarty_workspace && source install/setup.zsh
```

---

## Überblick

```text
Rosbag (/camera/image/raw, 2064x1544 mono8)
  │
  ├─ camera_preprocessing ──► /camera/image/undistorted
  │                       └─► /camera/image/bev  (Vogelperspektive, 800x640)
  │                                  │
  │                                  ▼
  │                         parking_perception_node
  │                           ├─ ParkingNet  (detector: 'parking_net')
  │                           └─ YOLO pose   (detector: 'yolo_pose')
  │                                  │
  │                                  ▼
  │                         /parking/debug_image  (Parkplätze eingezeichnet)
  │
  └─ tools/extract_frames.py ──► PNG pro Bag ──► CVAT ──► annotations.xml ──► Training ──► models/*.pt
```

Ein Parkplatz ist immer ein **Polygon mit genau 4 Ecken**. Die Ecken werden beim Laden in eine feste Reihenfolge gebracht: unten-links, unten-rechts, oben-rechts, oben-links (in Bildkoordinaten). Beide Modelle sagen die Ecken in dieser Reihenfolge voraus.

Klassen (Labels im CVAT-Projekt, Reihenfolge = Klassen-ID):

| ID | Label |
|---|---|
| 0 | `Parking_90_free` |
| 1 | `Parking_90_besetzt` |
| 2 | `Parking_90_gesperrt` |
| 3 | `Parking_Parallel_Frei` |
| 4 | `Parking_Parallel_Besetzt` |
| 5 | `Parking_Parallel_Gesperrt` |

Auf der Strecke liegen **90°-Parkplätze nur links** und **parallele Parkplätze nur rechts**. Deshalb ist das horizontale Spiegeln als Augmentierung in beiden Trainings ausgeschaltet.

---

## 1. Rosbag abspielen und Bilder ansehen

Jeder Befehl läuft in einem eigenen Terminal:

```bash
# Kamera-Vorverarbeitung (Entzerrung + Vogelperspektive)
ros2 launch camera_preprocessing camera_preprocessing.launch.py

# Bilder ansehen
/usr/bin/python3 /opt/ros/jazzy/lib/rqt_image_view/rqt_image_view

# Bag abspielen (Pfade mit Klammern immer in Anführungszeichen)
ros2 bag play /home/smartrollerz/rosbags/rosbag2_2026_10_28-13_36_43 --loop
```

In `rqt_image_view` wählst du oben das Topic aus: `/camera/image/bev` für die Vogelperspektive oder `/camera/image/undistorted` für das entzerrte Kamerabild.

Starte `rqt_image_view` mit `/usr/bin/python3`. Das Python aus pyenv hat kein PyQt5, deshalb bricht ein einfaches `rqt_image_view` mit einem ImportError ab.

Die Parking-Node liest standardmäßig die Vogelperspektive. Um sie auf das entzerrte Bild umzustellen, änderst du in `config/ros_params.yaml` zwei Werte:

```yaml
image_subscriber: "/camera/image/undistorted"
birds_eye: false
```

Optionen für `ros2 bag play`:
- `--rate 0.5` spielt halb so schnell ab.
- Die Leertaste pausiert.
- Die Pfeiltaste rechts geht ein Bild weiter.

---

## 2. Kamera kalibrieren

Die Vogelperspektive hängt an der Kalibrierung, also an der Position und dem Winkel der Kamera. **Schnelltest:** Laufen die Fahrspur- und Parkplatzlinien in der Vogelperspektive schräg aufeinander zu statt parallel, passt die Kalibrierung nicht zum Auto.

`camera_preprocessing` kalibriert aus **einem Foto**, auf dem ein Schachbrett flach vor dem Auto liegt (8×6 innere Ecken). Die Lage des Schachbretts gibt `world_points` in der Config von `camera_preprocessing` vor.

```bash
# 1. Alte Kalibrierung sichern
cp install/camera_preprocessing/share/camera_preprocessing/config/calib.bin ~/calib_2026-10-22.bin

# 2. Mit dem Schachbrett-Foto kalibrieren (beendet sich danach selbst)
ros2 launch camera_preprocessing camera_calibration.launch.py \
    chessboard_path:=/home/smartrollerz/rosbags/chessboard_2026-10-28.png
```

Am Ende sollte „Camera calibrated successfully.“ erscheinen. Mit `debug:=true` zeigt die Node die gefundenen Ecken und die neue Vogelperspektive. Um zurückzugehen, kopierst du die Sicherung wieder an dieselbe Stelle.

**Wichtig:**
- Nach einer neuen Kalibrierung sieht jedes Bild in der Vogelperspektive anders aus. Die Bilder musst du neu extrahieren, alte Annotationen und Modelle passen nicht mehr. Deshalb **erst kalibrieren, dann annotieren**.
- Die Rohbilder im Bag bleiben gültig. Nur die extrahierten Bilder müssen neu erzeugt werden.
- Die Kalibrierung muss zu den Bags passen: Das Schachbrett-Foto sollte vom selben Tag und Aufbau stammen.

---

## 3. Bilder aus Rosbags extrahieren

`tools/extract_frames.py` liest `/camera/image/raw` direkt aus dem Bag und verarbeitet die Bilder genau wie `camera_preprocessing`. Dafür nutzt es die installierte `calib.bin`. Ein laufendes ROS-System brauchst du nicht.

```bash
# Ein Bag, jedes 5. Bild, Vogelperspektive
python3 src/parking/tools/extract_frames.py \
    /home/smartrollerz/rosbags/rosbag2_2026_10_28-13_36_43 \
    src/parking/data/frames_v2/rosbag2_2026_10_28-13_36_43 --every 5

# Alle Bags, jeweils in einen eigenen Unterordner
for bag in /home/smartrollerz/rosbags/rosbag2_*; do
    python3 src/parking/tools/extract_frames.py "$bag" \
        "src/parking/data/frames_v2/$(basename "$bag")" --every 5
done
```

Optionen:

| Option | Bedeutung |
|---|---|
| `--every N` | Speichert nur jedes N-te Bild. Aufeinanderfolgende Bilder sind fast gleich. |
| `--view bev` / `--view undistorted` | Vogelperspektive (Standard) oder entzerrtes Bild |
| `--topic` | Anderes Kamera-Topic |

Die Bilder heißen `<bagname>_<framenummer>.png`. Daran erkennen die Notebooks, aus welchem Bag ein Bild stammt, und teilen Training und Validierung nach Bags auf. Benenne die Dateien deshalb nicht um.

Die lange Ausgabe am Anfang ist die Kalibrierung von `camera_preprocessing`, kein Fehler.

---

## 4. CVAT lokal mit Docker

Die Web-Version von CVAT brach beim Hochladen mit „Bad Gateway“ ab. Lokal liest CVAT die Bilder direkt aus dem Ordner `data/frames_v2` („Connected file share“), ein Upload ist dann nicht nötig.

### Einmalig installieren

```bash
git clone https://github.com/cvat-ai/cvat ~/cvat
cd ~/cvat
git checkout $(git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
```

Die Datei `~/cvat/docker-compose.override.yml` bindet den Bilderordner als Share ein:

```yaml
services:
  cvat_server:
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_import:
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_export:
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_annotation:
    volumes:
      - cvat_share:/home/django/share:ro
  cvat_worker_chunks:
    volumes:
      - cvat_share:/home/django/share:ro

volumes:
  cvat_share:
    driver_opts:
      type: none
      device: /home/smartrollerz/smarty_workspace/src/parking/data/frames_v2
      o: bind
```

```bash
cd ~/cvat
docker compose up -d                 # Beim ersten Mal werden die Images geladen (einige Minuten)
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Danach öffnest du **http://localhost:8080**. Offiziell unterstützt CVAT nur Chrome.

### Starten, stoppen, Share-Ordner wechseln

```bash
cd ~/cvat && docker compose up -d     # starten
cd ~/cvat && docker compose down      # stoppen, Annotationen bleiben erhalten
```

Verwende **niemals** `docker compose down -v`. Das löscht alle Projekte und Annotationen.

So wechselst du den Share-Ordner:
1. Trag in der Override-Datei unter `device:` den neuen Pfad ein.
2. Lösch das alte Share-Volume, denn Docker übernimmt den neuen Pfad sonst nicht:

```bash
cd ~/cvat
docker compose down
docker volume rm cvat_cvat_share
docker compose up -d
```

Dabei geht nichts verloren. Das Volume ist nur der Verweis auf den Ordner.

### Projekt und Tasks anlegen

1. Leg **ein Projekt** an, mit den sechs Labels aus der Tabelle oben, jeweils vom Typ **Polygon**. Alle Tasks gehören in dieses Projekt, damit die Labels überall gleich sind.
2. Leg **pro Bag einen Task** an. Wähl unter „Select files“ den Reiter **„Connected file share“** und dort den Ordner des Bags.
3. Die Task-ID steht in der URL, z. B. `localhost:8080/tasks/4` → `4`.

---

## 5. Annotieren

Regeln:
- **Ein Polygon mit genau 4 Punkten pro Parkplatz.** Die Reihenfolge der Punkte ist egal, sie wird beim Laden sortiert.
- Das Label ist die Klasse. 90°-Parkplätze liegen nur links, parallele nur rechts.
- **Annotiere jedes Bild des Tasks vollständig.** Ein Bild ohne Polygon zählt im Training als „hier ist kein Parkplatz“. Ein vergessener Parkplatz bringt dem Modell also etwas Falsches bei.
- Teilweise sichtbare Parkplätze annotierst du nur, wenn alle 4 Ecken im Bild liegen oder sich gut schätzen lassen. Annotierst du sie, dann einheitlich.
- Ein Bild, das ein Polygon mit 5 oder 6 Punkten oder ein unbekanntes Label enthält, wird beim Training **komplett übersprungen**. Das Notebook zeigt dazu eine Warnung mit Beispielen.

### Vorannotation mit einem trainierten Modell

Ab dem ersten Modell kann CVAT neue Tasks vorannotieren, und du korrigierst nur noch. Native Functions aus der CVAT-Doku gibt es in der Community-Version nicht. Stattdessen lässt `cvat-cli` das Modell lokal laufen und lädt die Polygone hoch.

```bash
pip install cvat-cli
cd ~/smarty_workspace

# Mit ParkingNet
cvat-cli --server-host http://localhost --server-port 8080 --auth <benutzer> \
    task auto-annotate <task-id> \
    --function-file src/parking/tools/cvat_function.py \
    -p model_path=str:src/parking/models/parking_net.pt \
    --conf-threshold 0.3 --allow-unmatched-labels

# Mit YOLO pose
cvat-cli --server-host http://localhost --server-port 8080 --auth <benutzer> \
    task auto-annotate <task-id> \
    --function-file src/parking/tools/cvat_function_yolo.py \
    -p model_path=str:src/parking/models/yolo_pose.pt \
    --conf-threshold 0.3 --allow-unmatched-labels
```

Danach kontrollierst du jedes Bild:
- Klasse prüfen
- Ecken verschieben
- Fehlende Parkplätze ergänzen
- Falsche Parkplätze löschen

---

## 6. Exportieren und Datensatz anlegen

In CVAT exportierst du über Task → **Export task dataset** → Format **„CVAT for images 1.1“**. Lege einen neuen Ordner pro Datensatzversion an:

```text
src/parking/data/dataset_v4/
├── annotations.xml                    (bei mehreren Tasks: annotations_<bag>.xml, alle annotations*.xml werden geladen)
└── rosbag2_2026_10_28-13_32_18/       (Bilder des Bags, Kopie aus frames_v2)
    ├── rosbag2_2026_10_28-13_32_18_000000.png
    └── ...
```

Im Notebook stellst du den Datensatz über `DATA_DIR = PACKAGE_DIR / "data" / "dataset_v4"` ein.

### Datensatz prüfen

Die zweite Code-Zelle beider Notebooks gibt aus:
- Anzahl der Bilder und Parkplätze
- Anzahl pro Klasse
- Aufteilung in Training und Validierung
- gegebenenfalls eine Warnung zu übersprungenen Bildern

So prüfst du die Ausgabe:
- **Übersprungene Bilder:** In der Warnung stehen Bildname, Label und Punktanzahl. Korrigier diese Bilder in CVAT und exportier neu.
- **Klasse mit 0 Beispielen:** Kann das Modell nicht lernen. Es braucht mehr Daten dieser Klasse.
- **Falsche Seite:** Steht ein 90°-Label rechts, ist es ein Annotationsfehler.

Die Labels aller XML-Dateien eines Datensatzes müssen gleich sein. Das ist automatisch der Fall, wenn alle Tasks im selben Projekt liegen. Sonst bricht das Notebook mit `Different labels in ...` ab.

---

## 7. Training

### Installation

```bash
# PyTorch für CPU. torch und torchvision IMMER zusammen aus demselben Index installieren
pip install "torch==2.14.1+cpu" "torchvision==0.29.2+cpu" --index-url https://download.pytorch.org/whl/cpu

# Notebook-Abhängigkeiten
pip install -r src/parking/requirements-training.txt

# Nur für YOLO, OHNE -U (sonst springt numpy auf 2.x und cv_bridge funktioniert nicht mehr)
pip install -r src/parking/requirements-yolo.txt
```

`torch 2.15` funktioniert nicht mit Ultralytics, weil es noch keine passende `torchvision` gibt. Der Fehler lautet dann `operator torchvision::nms does not exist`.

Prüfen:

```bash
python3 -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
# Erwartet: 2.14.1+cpu 0.29.2+cpu 1.26.x
```

### Modell A: ParkingNet (eigenes kleines Netz)

Das Notebook liegt unter `src/parking/notebooks/train_parking_net.ipynb`. Das Netz arbeitet im CenterNet-Stil:
- eine Heatmap pro Klasse für die Mitte des Parkplatzes
- 8 Werte für die Abstände der 4 Ecken zur Mitte
- Eingabe 400×320 Pixel

Trainiert wird 50 Epochen lang. Das beste Modell landet in `models/parking_net.pt`. Bei rund 100 Bildern dauert das auf der CPU wenige Minuten.

### Modell B: YOLO pose (Ultralytics)

Das Notebook liegt unter `src/parking/notebooks/train_yolo_pose.ipynb`. Für YOLO ist jeder Parkplatz ein Objekt mit Klasse und Rahmen, seine 4 Ecken sind Keypoints. Das Notebook:
1. exportiert die CVAT-Annotationen nach `data/yolo/<datensatz>/`
2. trainiert ab dem vortrainierten `yolo11n-pose.pt`
3. legt die Ergebnisse in `runs/pose/parking_<datensatz>/` ab, darunter `results.png` mit den Kurven
4. kopiert das beste Modell nach `models/yolo_pose.pt`

Auf der CPU dauert das deutlich länger als beim ParkingNet, etwa eine Stunde für 100 Epochen. Das Training stoppt früher, wenn sich 30 Epochen lang nichts verbessert.

### Wichtig beim Arbeiten mit den Notebooks

- **Nach Änderungen an Code oder Daten** den Kernel neu starten und alles neu ausführen: oben **Restart**, dann **Run All**. Ein laufender Kernel behält einmal importierten Code. Ein Modell kann sonst unbemerkt mit alten Daten trainiert werden.
- Wurde ein Notebook außerhalb von VS Code geändert, schließt du den Tab ohne zu speichern („Don't Save“) und öffnest ihn neu.
- Die Validierung besteht aus ganzen Bags. Gibt es nur einen Bag, wird zufällig aufgeteilt, und die Validierungswerte sind dann zu optimistisch.

---

## 8. Modell als Node auf einem Rosbag testen

Welches Modell die Node verwendet, steht in `src/parking/config/model.yaml`:

```yaml
detector: 'yolo_pose'      # oder 'parking_net'
score_threshold: 0.3       # Mindest-Score einer Erkennung
```

Das Paket ist mit `--symlink-install` gebaut. Ein neu trainiertes Modell oder eine geänderte `model.yaml` gilt deshalb sofort. Nur die **Node musst du neu starten**. Neu bauen musst du nur, wenn neue Dateien dazukommen:

```bash
colcon build --symlink-install --packages-select parking
```

Test: Jeder Befehl in einem eigenen Terminal, jeweils nach `source install/setup.zsh`. Nimm am besten einen Bag, der nicht im Training war.

```bash
ros2 launch camera_preprocessing camera_preprocessing.launch.py
ros2 launch parking parking_perception.launch.py debug:=True
/usr/bin/python3 /opt/ros/jazzy/lib/rqt_image_view/rqt_image_view /parking/debug_image
ros2 bag play /home/smartrollerz/rosbags/rosbag2_2026_10_28-13_36_43 --loop
```

**Debug-Bild:**
- Rot umrandet: ein erkannter Parkplatz.
- Grüner Punkt mit Text: die Mitte des Parkplatzes, die Klasse und der Score, z. B. `Parking_Parallel_Frei 0.74`.
- Blauer Punkt: die erste Ecke (unten-links).

**Log der Node:**
- Beim Start erscheint `Loaded model ...` (ParkingNet) bzw. `Loaded YOLO model ...`.
- Erscheint stattdessen `No trained model found`, fehlt die Modelldatei.
- Die Zeiten pro Bild stehen unter `total` und `prediction`. Das ParkingNet braucht auf dem Laptop etwa 35 ms pro Bild.

**Vergleich der beiden Modelle:** Auf demselben Bag einmal mit `detector: 'parking_net'` und einmal mit `'yolo_pose'` laufen lassen. Achte auf:
- übersehene Parkplätze
- falsche Klassen
- falsche Seite
- wie genau die Ecken sitzen
- die `prediction`-Zeit

---

## 9. Was in Git landet

Diese Dateien landen in Git:
- der Code
- die Notebooks
- die Configs
- die trainierten Modelle in `models/` (`parking_net.pt`, `yolo_pose.pt`)

- der bereinigte Datensatz `data/dataset_v4/` (Bilder + `annotations.xml`), damit er auf anderen Rechnern (z. B. zum Training auf einem Mac) mitgeklont wird

Einen neuen Datensatz nimmst du mit einer weiteren Zeile `!/data/dataset_vN/` in der `.gitignore` auf. Nimm nur fertige, bereinigte Datensätze auf, jede Version bleibt dauerhaft in der Git-Historie.

Diese Dateien sind per `.gitignore` ausgeschlossen:
- der Rest von `data/`: Frames, ältere Datensätze, CVAT-Uploads, YOLO-Export
- `runs/`: Trainingsläufe von Ultralytics
- Rosbags (`*.mcap`, `*.db3`)
- ZIP-Dateien
- `.pt`-, `.pth`- und `.onnx`-Dateien außerhalb von `models/`

Branches:
- `MNIST/Model`: ParkingNet
- `YOLO/Pose`: zusätzlich YOLO pose

---

## 10. Häufige Fehler

| Fehler | Ursache / Lösung |
|---|---|
| `rqt_image_view`: ImportError PyQt5 | Mit `/usr/bin/python3 /opt/ros/jazzy/lib/rqt_image_view/rqt_image_view` starten |
| `zsh: no matches found` / Pfadfehler beim Bag | Pfad mit Klammern in Anführungszeichen setzen. Platzhalter wie `<pfad>` durch echte Pfade ersetzen. |
| CVAT: „Bad Gateway“ beim Upload | Lokales CVAT mit „Connected file share“ verwenden (Abschnitt 4) |
| CVAT zeigt den neuen Share-Ordner nicht | `docker volume rm cvat_cvat_share`, siehe Abschnitt 4 |
| YAML in das Terminal eingefügt → zsh-Fehler | YAML-Blöcke sind Dateiinhalte, keine Befehle |
| `vimbax_camera` nicht gefunden | Gibt es nur auf dem Auto. Am Laptop mit einem vorhandenen Schachbrett-Foto kalibrieren. |
| Modell erkennt Klassen auf der falschen Seite | Annotationsfehler prüfen. Flip-Augmentierung muss aus sein. Kernel neu starten, damit nicht mit alten Daten trainiert wird. |
| `operator torchvision::nms does not exist` | `torch` und `torchvision` passen nicht zusammen, siehe Abschnitt 7 |
| Bilder werden beim Training übersprungen | Polygon ohne genau 4 Punkte oder unbekanntes Label. Steht in der Warnung, in CVAT korrigieren. |
