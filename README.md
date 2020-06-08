# collecting_with_dorna

## Installation
ROS Melodic installieren:
http://wiki.ros.org/melodic/Installation/Ubuntu

collecting-with-dorna Repositoty ins Catkin-Workspace herunterladen:
```bash
cd ~/catkin_ws/src
git clone https://github.com/lennarttoenjess/collecting_with_dorna.git
```

dorna Repository herunterladen und installieren:
```bash
cd collecting_with_dorna
git clone https://github.com/dorna-robotics/dorna
pip3 install setuptools
cd dorna
sudo python3 setup.py install
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

IDS-Kameratreiber für die UI-3880LE-C-HQ herunterladen auf:
https://de.ids-imaging.com/download-details/AB02648.html
Verzeichnis enpacken
im entpackten Verzeichnis: 
```bash
sudo ./ueye_4.93.0.989_amd64.run
```

Pyueye installieren:
```bash
pip3 install pyueye
```

OpenCV installieren:
```bash
pip3 install opencv-python
```

Pyzbar installieren:
```bash
sudo apt update
sudo apt-get install zbar-tools
pip3 install pyzbar
```

## Anwendung
### Befehle an den Roboter geben
Dorna über USB mit dem Computer verbinden

```bash
cd ~/catkin_ws/src/collecting_with_dorna/src
python3 move_dorna.py
```

### Kamera intrinsisch kalibrieren
Kamera über USB mit dem Computer verbinden

```bash
cd ~/catkin_ws/src/collecting_with_dorna/src
python3 camera_intrinsisc_calibration.py
```

### Positionsbestimmung der Objekte
Kamera über USB mit dem Computer verbinden

```bash
cd ~/catkin_ws/src/collecting_with_dorna/src
python3 camera.py
```
oder
```bash
cd ~/catkin_ws/src/collecting_with_dorna/src
python3 camera_multible_colors.py
```
für Positionsbestimmung von Objekten mit verschiedenen Farben

### Bälle automatisch einsammeln
Kamera und Dorna über USB mit dem Computer verbinden

```bash
roscore
```
in einem neuem Terminal:

```bash
cd ~/catkin_ws
source devel/setup.bash
cd src/collecting_with_dorna/src
python3 collecting_with_dorna_publisher.py
```

in einem neuem Terminal:

```bash
cd ~/catkin_ws
source devel/setup.bash
cd src/collecting_with_dorna/src
python3 collecting_with_dorna_subscriber.py
```
