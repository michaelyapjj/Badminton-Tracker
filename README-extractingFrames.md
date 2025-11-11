Extracting frames from videos on windows
We are doing this to label each frame to train YOLO.

Have ffmpeg downloaded

Create a folder called frames. This code extracts the frames from your video to the frames file.
-q:v 1 → high image quality
-r 10 → 10 frames per second
```
ffmpeg -i "C:\Users\<Your Name>\Videos\<Your Video>.mov" -q:v 1 -r 10 "C:\Users\<Your Name>\Project\frames\frame_%04d.jpg"
```
