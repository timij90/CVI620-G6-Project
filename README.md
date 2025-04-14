# Self-Driving Car Simulation Using CNN

> **Final Project — CVI620 (Winter 2025)**  
> Developed for the Udacity Self-Driving Car Simulator using Python, OpenCV, and TensorFlow.

---

## Project Overview

This project implements a convolutional neural network (CNN) to control the steering angle of a simulated self-driving car. The model predicts steering angles in real-time using images from the car’s front-facing camera, enabling autonomous navigation along a predefined track.

We use the Udacity self-driving car simulator for data collection and testing. The CNN model was trained on images captured during manual driving and evaluated within the simulator to test its real-world performance in real-time.

---

## Technologies Used

- **Python**
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy**
- **Flask + SocketIO**
- **Pandas**
- **Matplotlib**

---

## Setup Instructions

1. **Create a Virtual Environment:**

   ```bash
   python -m venv car-env
   source car-env/bin/activate  # On Windows: car-env\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r package_list.txt
   ```

3. **Download the Udacity Simulator:**
   - (Simulator Link Version 2 Link)[https://github.com/udacity/self-driving-car-sim?tab=readme-ov-file#:~:text=Linux%20Mac%20Windows-,Version%201%2C%2012/09/16,-Linux%20Mac%20Windows]

---

## Data collection

1. Launch the simulator and select **Training Mode**.
2. Manually drive the car using your keyboard or mouse.
3. Save the data by clicking Recording, which generates:
  - A folder of images (IMG/)
  - A CSV file (driving_log.csv)

Drive several laps in both directions to balance your dataset.

---

## Training the Model

Run the following to preprocess and train the model:

```bash
python train_model.py
```
The final model is saved as `model.h5`. A loss graph (`training_plot.png`) is also generated

---

## Testing the Model

1. Start the simulator in **Autonomous Mode**.
2. Run the testing script:
   ```bash
   python TestSimulation.py
   ```
3. The car will start moving using predictions from your trained model.

---

## Challenges & Solutions


---

## Demo
(Link to demo video)[]
