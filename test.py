import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import glob
import os

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))



# Function to download and extract the dataset
def get_data_extract():
    if "dataset" in os.listdir():
        print("Dataset already exists")
    else:
        print("Downloading the data...")
        !wget -O food-data.zip https://www.kaggle.com/api/v1/datasets/download/trolukovich/food11-image-dataset
        print("Dataset downloaded!")
        print("Extracting data...")
        os.makedirs("dataset", exist_ok=True)
        !unzip -q food-data.zip -d dataset
        print("Extraction done!")



get_data_extract()

path = glob.glob("dataset/*/*/*.jpg")
label = [i.split(".")[0].split("/")[-2] for i in path]

# Create image augmentation pipeline
image_width, image_height = 224, 224  # Adjust dimensions for EfficientNet
aug = tf.keras.Sequential([
    tf.keras.layers.Resizing(image_width, image_height),
    tf.keras.layers.Rescaling(1./255)  # Normalize pixel values
])


# Function to load and preprocess images
def load_image(path, label, aug):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    # Apply augmentation
    image = aug(image)
    return image, label

# Create a TensorFlow Dataset pipeline
ds = tf.data.Dataset.from_tensor_slices((path, label))
ds = ds.shuffle(buffer_size=len(path), reshuffle_each_iteration=False)  # Prevent data leakage
ds = ds.map(lambda x, y: load_image(x, y, aug)).batch(32).prefetch(tf.data.AUTOTUNE)

# Split dataset into training, validation, and test sets
train_size = int(0.7 * len(path))
val_size = int(0.2 * len(path))
test_size = len(path) - train_size - val_size
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size).take(val_size)
test_ds = ds.skip(train_size + val_size).take(test_size)

# Display a sample from the dataset
print("Sample from the dataset:", next(iter(train_ds))[1])

# Define the model architecture using EfficientNet
def build_model(img_size=(224, 224, 3), num_classes=11):  # Adjust num_classes as per dataset
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=img_size)
    base_model.trainable = False  # Freeze the base model for feature extraction

    # Add classification head
    inputs = tf.keras.Input(shape=img_size)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model

# Compile and Train the Model
def train_model(model, train_data, val_data, learning_rate=0.001, epochs=10):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks for training
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=3)
    ]

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

# Build and train the model
model = build_model()
history = train_model(model, train_ds, val_ds)

# Fine-tuning: Unfreeze some layers and retrain
def fine_tune_model(model, learning_rate=0.0001, fine_tune_at=100):
    model.trainable = True  # Unfreeze the entire model
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False  # Keep earlier layers frozen

    # Compile and retrain
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train with reduced learning rate
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=3)
        ]
    )
    return history

# Fine-tune the model
fine_tune_history = fine_tune_model(model)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model
model.save("efficientnet_food_classifier.h5")
print("Model saved as efficientnet_food_classifier.h5")
