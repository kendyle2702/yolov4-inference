import argparse
import time
import glob
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=float, default=0.5,
        help="minimum probability to filter weak detections")

    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf
    NMS_THRESHOLD = 0.4

    weights = "pretrained/yolov4.weights"
    labels = "pretrained/labels.txt"
    cfg = "pretrained/yolov4.cfg"
    input_folder = "images"
    output_folder = "results"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

    lbls = list()
    with open(labels, "r") as f:
        lbls = [c.strip() for c in f.readlines()]

    COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer = net.getLayerNames()
    layer = [layer[i - 1] for i in net.getUnconnectedOutLayers()]

    def detect(image, nn):
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255, (640, 640), swapRB=True, crop=False)
        nn.setInput(blob)
        start_time = time.time()
        layer_outs = nn.forward(layer)
        end_time = time.time()

        boxes = list()
        confidences = list()
        class_ids = list()

        for output in layer_outs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (center_x, center_y, width, height) = box.astype("int")

                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(lbls[class_ids[i]], confidences[i])
                cv2.putText(
                    image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                label = "Inference Time: {:.2f} s".format(end_time - start_time)
                cv2.putText(
                    image, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
                )
        
        return image

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"{image_file}")
        
        print(f"Processing {image_file}...")
        
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image {image_file}")
            continue
            
        result = detect(image, net)
        
        cv2.imwrite(output_path, result)
        print(f"Saved result to {output_path}")

if __name__ == "__main__":
    main()
