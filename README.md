# collecting_with_dorna

## Installation
ROS Melodic installieren:
http://wiki.ros.org/melodic/Installation/Ubuntu

```bash
cd ~/catkin_ws/src
```

collecting-with-dorna Repositoty im Catkin Workspace herunterladen:
```bash
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


