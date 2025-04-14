import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random

# DATA
def augment_image(image, steering_angle):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def load_data(csv_file):
    data_df = pd.read_csv(csv_file, header=None)
    data_df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    def fix_path(p): return os.path.join('IMG', os.path.basename(p.strip().replace("\\\\", "/")))

    data = []

    for index, row in data_df.iterrows():
        center = fix_path(row['center'])
        angle = row['steering']

        if os.path.exists(center):
            data.append((center, angle))
            if abs(angle) > 0.1:
                data.extend([(center, angle)] * 10)  
            if abs(angle) > 0.2:
                data.extend([(center, angle)] * 10)      
            if abs(angle) > 0.4:
                data.extend([(center, angle)] * 10)  
            if abs(angle) > 0.5:
                data.extend([(center, angle)] * 10)
            if abs(angle) > 0.7:
                data.extend([(center, angle)] * 10)
            if abs(angle) > 0.9:
                data.extend([(center, angle)] * 10)        

    df = pd.DataFrame(data, columns=['image', 'steering'])
    df = df[df['steering'].notnull()]

    turning = df[abs(df['steering']) > 0.05]
    straight = df[abs(df['steering']) <= 0.05].sample(frac=0.2, random_state=42)
    df = pd.concat([turning, straight]).sample(frac=1.0).reset_index(drop=True)

    image_paths = df['image'].values
    steerings = df['steering'].values
    return image_paths, steerings


def preprocess_image(image):
    image = image[60:135, :, :] 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255.0
    return image

def load_batch(image_paths, steerings, augment=False):
    images = []
    angles = []
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        angle = steerings[i]

        if augment:
            image, angle = augment_image(image, angle)

        image = preprocess_image(image)
        images.append(image)
        angles.append(angle)
    return np.array(images), np.array(angles)

# MODEL
def self_model():
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1e-4))
    return model

# EVALUATION
def train_model():
    csv_file = 'driving_log.csv'
    image_paths, steerings = load_data(csv_file)
    image_paths_train, image_paths_val, steerings_train, steerings_val = train_test_split(
        image_paths, steerings, test_size=0.2, random_state=42
    )

    X_train, y_train = load_batch(image_paths_train, steerings_train, augment=True)
    X_val, y_val = load_batch(image_paths_val, steerings_val, augment=False)

    model = self_model()
    
    early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,          
    restore_best_weights=True
)
    
    print("🚗 Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=40,
        validation_data=(X_val, y_val),
        batch_size=64,
        shuffle=True,
        callbacks=[early_stop]
    )

    model.save('model.h5')
    print("✅ Model saved as model.h5")

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('training_plot.png')
    plt.show()

if __name__ == '__main__':
    train_model()
