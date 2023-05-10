import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from config import MODEL_DIR, MODEL_NAME
from data_loader import load_data

def evaluate_model():
    """
    Evaluate the emotion recognition model on the test set.
    """
    # Load data
    _, _, test_generator = load_data()

    # Load the trained model
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model = load_model(model_path)

    # Make predictions on the test set
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate the confusion matrix and classification report
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred_classes)
    cr = classification_report(y_true, y_pred_classes, target_names=class_labels)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

if __name__ == "__main__":
    evaluate_model()
    print("Evaluation completed.")
