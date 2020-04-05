# collecting-with-dorna

## Installation
ROS Melodic installieren:
http://wiki.ros.org/melodic/Installation/Ubuntu

collecting-with-dorna Repositoty herunterladen:
```bash
git clone https://github.com/lennarttoenjess/collecting-with-dorna.git
```

dorna Repository herunterladen und installieren:
```bash
cd collecting-wit-dorna
git clone https://github.com/dorna-robotics/dorna
cd dorna
pip3 install setuptools
sudo python3 setup.py install
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


