# Downloading Python and setting up YOLOv5n environment

Installing python on host and your VM:
Download python 3.14: https://www.python.org/downloads/ 

We download on both because the Recommended Workflow is as follows:
1. Windows Host – Development & Training
- Train YOLOv5n on your shuttlecock dataset (fast GPU if available).
- Test detection on local videos until it’s stable.
- Export the trained model to ONNX or TFLite (best.onnx or best.tflite).

2. Linux VM – Deployment Simulation
- Re-create the Beagle’s environment:
```
sudo apt install python3-venv python3-pip opencv-python
pip install torch torchvision ultralytics
```
- Verify that your model loads and runs inference here using CPU.
- Fix any library/version issues before touching the Beagle board.

3. BeagleY-AI Target – Real-Time Execution
- Copy the validated model + script to the board (scp or USB).
- Install only lightweight runtime libs (OpenCV, TIDL/TensorRT, etc.).
- Connect the iPhone camera stream and run your live detection/speed tracker.
- Profile FPS and latency; adjust quantization if it’s too slow.
  
On VM after installing the pythone package: 
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

#Troubleshoot
# If during installation it says out of space
sudo mount -o remount,size=6G /tmp
# and re-run 
pip install -r requirements.txt --prefer-binary
```
