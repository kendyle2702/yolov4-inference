# YOLOV4  -  You Only Look Once 

## Description
This is a project to run inference of pretrained YOLOV4 models with opencv.

![Image](https://github.com/user-attachments/assets/5ca78dd7-df3e-422d-b0ca-ebb2170af35d)
## 🚀 Quick Start

### Requirements
I recommend you to use python >= 3.9 to run project.

### **1️⃣ Clone the Project**

Clone with HTTPS
```bash
  git clone https://github.com/kendyle2702/yolov4-inference.git
  cd yolov4-inference
```
Clone with SSH
```bash
  git clone git@github.com:kendyle2702/yolov4-inference.git
  cd yolov4-inference
```

### **2️⃣ Install Library**
```bash
  pip install -r requirements.txt
```

### **3️⃣ Download Pretrained Model**

Download YOLOv4 [pretrained weight](https://github.com/WongKinYiu/PyTorch_YOLOv4/releases/download/weights/yolov4.weights). 

Move pretrained weight to ```pretrained``` folder. 


### **4️⃣ Run Inference**
Make sure you have put the images you need to inference into the ```images``` folder.
```bash
  python main.py --conf {default 0.5}
```
The image inference results will be in the ```results``` folder.
