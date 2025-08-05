import argparse
import pathlib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from dataload import load_dataset
from model import build_cnn_bilstm


def parse_args():
    ap = argparse.ArgumentParser(description="Train SM-CBNet.")
    ap.add_argument("--data", required=True, help="Path to CSV dataset.")
    ap.add_argument("--target", required=True, help="Target column name.")
    ap.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    ap.add_argument("--batch", type=int, default=64, help="Batch size.")
    ap.add_argument("--no_oversample", action="store_true",
                    help="Disable RandomOverSampler.")
    return ap.parse_args()


def main():
    args = parse_args()

    (X_train, X_val,
     y_train, y_val,
     n_classes, input_shape,
     class_weights) = load_dataset(
        csv_path=args.data,
        target_column=args.target,
        oversample=not args.no_oversample
    )

    model = build_cnn_bilstm(input_shape, n_classes)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weights,
        verbose=2
    )

    # Evaluate
    y_pred = model.predict(X_val)
    if n_classes == 2:
        y_pred_labels = (y_pred > 0.5).astype(int).flatten()
    else:
        y_pred_labels = y_pred.argmax(axis=1)

    print(confusion_matrix(y_val, y_pred_labels))
    print(classification_report(y_val, y_pred_labels))


if __name__ == "__main__":
    main()