# Badminton shuttle speedometer

Installing python on host and your VM:
Download python 3.14: https://www.python.org/downloads/ 
On VM: 
```
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
    liblzma-dev tk-dev wget

cd ~/Downloads/Python-3.14.0
make clean
./configure --enable-optimizations
make -j$(nproc)
sudo make install

#To verify
python3 --version 
python3 -m ensurepip --upgrade
```
Set up environment:
```
# Inside your home directory or project folder
python3 -m venv yolov5env
source yolov5env/bin/activate

pip install --upgrade pip wheel setuptools

git clone https://github.com/ultralytics/yolov5.git
cd yolov5

pip install -r requirements.txt
#If PyTorch fails because of missing build tools, install them:
sudo apt install libopenblas-dev libomp-dev

#Verify
python detect.py --source data/images/bus.jpg --weights yolov5n.pt

#To save environment
pip freeze > requirements_locked.txt

#To deactivate environment
deactivate
```
