import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler


def load_dataset(csv_path: str,
                 target_column: str,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 oversample: bool = True):
    """
    Loads a CSV dataset, splits into train/val, applies scaling and (optional) oversampling.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    target_column : str
        Name of the target column.
    test_size : float, default 0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default 42
        Random seed for reproducibility.
    oversample : bool, default True
        Whether to apply RandomOverSampler to balance classes.

    Returns
    -------
    X_train, X_val, y_train, y_val : np.ndarray
        Prepared numpy arrays ready for model training.
    n_classes : int
        Number of unique classes (for output layer sizing).
    input_shape : tuple
        Shape to expect for model input (timesteps, features).
    class_weights : dict
        Class weights dict that can be fed directly to `model.fit`.
    """

    df = pd.read_csv(csv_path)
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # feature scaling (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # reshape to (samples, timesteps, features)
    # Here we treat each row as a 1‑D sequence (timesteps= X.shape[1])
    # Adjust this if your data already has a temporal dimension.
    timesteps = X_train.shape[1]
    X_train = X_train.reshape(-1, timesteps, 1)
    X_val = X_val.reshape(-1, timesteps, 1)

    # oversampling (only on training set)
    class_weights = None
    if oversample:
        ros = RandomOverSampler(random_state=random_state)
        X_train_rs, y_train_rs = ros.fit_resample(
            X_train.reshape(X_train.shape[0], -1), y_train)
        X_train = X_train_rs.reshape(-1, timesteps, 1)
        y_train = y_train_rs
    else:
        # compute class weights for imbalanced data
        uniq = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=uniq, y=y_train)
        class_weights = {int(k): float(v) for k, v in zip(uniq, cw)}

    n_classes = len(np.unique(y))
    input_shape = X_train.shape[1:]  # (timesteps, features)

    return X_train, X_val, y_train, y_val, n_classes, input_shape, class_weights