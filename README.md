# BugHunt

### HexBug Multi‑Object Tracking Pipeline 🪲🎥

This project implements an **automated multi-object tracking (MOT) pipeline** for tracking **HexBug heads** across video frames.
The pipeline takes a **raw video** as input and outputs a **CSV file** containing the **head coordinates** and **persistent IDs** for each detected HexBug in every frame.

#### **Pipeline Overview**

1. **Video Preprocessing**

   * The raw video is split into individual frames.
   * The aspect ratio of the first frame determines whether to use **YOLOv8-S** or **YOLOv8-M** for detection.

2. **Detection (YOLOv8)**

   * YOLOv8 is used to detect HexBug heads.
   * It outputs bounding boxes in the format:
     $`x1`, `y1`, `x2`, `y2`, `confidence`$.

3. **Tracking (Modified SORT)**

   * Detections are passed to a customized **SORT** tracker:

     * Bounding boxes are converted into **Kalman filter states**.
     * IDs are assigned via **centroid-distance-based Hungarian matching** with **adaptive gating**.
     * **ID recycling** and **automatic ceiling enforcement** are used to maintain consistent track counts.

4. **Results**

   * The tracker outputs a CSV file containing IDs and coordinates per frame:
     $`frame`, `ID`, `x`, `y`$.

