# VPTAU2022_Project
This project is about stabilization, background subtraction, matting, and tracking. 
We get a shaky video of a person walking. We implement several algorithms throughout this project which eventually, output a stabilized video of that person walking, 
with a different background and rectangle around the person tracking it through the entire video.

![image](https://github.com/arielkantorovich/VPTAU2022_Project/assets/56262208/c7f3728a-84ec-48e2-a199-23fdf229f2c9)

Generally, the flow diagram we will implement is the following:
![image](https://github.com/arielkantorovich/VPTAU2022_Project/assets/56262208/65e60b6d-20ef-48ae-a2f6-561977103f60)

Blocks implementation are shown in blue, the videos are in red. 
First we will stabilize the input video to get: stabilize.avi. Next, we will subtract the background from the stabilized video to achieve: extracted.avi and binary.avi. 
Then we take the new background image and perform matting to output a matted video: matted.avi along with an alpha.avi video. 
The final phase is Tracking. We will track the subject walking in the video and put a rectangle around it. We get an OUTPUT.avi. We test our project on Ubuntu 16.04 machine.
