import tensorflow as tf
from src.config import EPOCHS, INITIAL_LR, FINE_TUNE_LR, MODEL_DIR, MODEL_NAME
from src.data_loader import load_data
from src.model import create_model

def train_model():
    """
    Train the emotion recognition model.
    """
    # Load data
    train_generator, val_generator, _ = load_data()

    # Create model
    model = create_model()

    # Train the top layers
    print("Training the top layers...")
    model.fit(train_generator,
              validation_data=val_generator,
              epochs=EPOCHS,
              verbose=2)

    # Fine-tune the base model
    print("Fine-tuning the base model...")
    base_model = model.layers[0]
    base_model.trainable = True

    # Recompile the model with a smaller learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=FINE_TUNE_LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator,
              validation_data=val_generator,
              epochs=EPOCHS,
              verbose=2)

    # Save the model
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))
    print(f"Model saved as {MODEL_NAME}")

if __name__ == "__main__":
    train_model()
    print("Training completed.")
