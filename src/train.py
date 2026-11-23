"""Train and evaluate an original MLP neural network with a group-held-out test set."""
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['title_length', 'title_words', 'has_number', 'has_question',
           'published_hour_utc', 'published_weekday_utc', 'log_duration']
CATEGORICAL = ['category']
FEATURES = NUMERIC + CATEGORICAL + ['title']


def train():
    tf.keras.utils.set_random_seed(42)
    df = pd.read_csv(ROOT / 'data/processed/videos_features.csv')
    df['title'] = df['title'].fillna('').astype(str)
    X, y = df[FEATURES], df['above_expected'].astype(int)
    groups = df['channel_id'].fillna(df['video_id']).astype(str)
    if groups.nunique() < 8:
        raise ValueError('Poucos canais distintos para separar treino e teste por canal.')
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(outer.split(X, y, groups))
    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]
    groups_trainval = groups.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=24)
    train_idx, val_idx = next(inner.split(X_trainval, y_trainval, groups_trainval))
    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
    if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError('Uma partição contém apenas uma classe. Colete mais vídeos/canais.')
    preprocessing = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ('title_tfidf', TfidfVectorizer(max_features=200, min_df=2, ngram_range=(1, 2), strip_accents='unicode', sublinear_tf=True), 'title'),
    ])
    train_array = preprocessing.fit_transform(X_train)
    val_array = preprocessing.transform(X_val)
    test_array = preprocessing.transform(X_test)
    # Comparação com um modelo linear treinado exatamente nos mesmos dados.
    linear = LogisticRegression(max_iter=1000, random_state=42)
    linear.fit(train_array, y_train)
    linear_probs = linear.predict_proba(test_array)[:, 1]
    linear_preds = (linear_probs >= 0.5).astype(int)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(train_array.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    history = model.fit(train_array, y_train.to_numpy(), validation_data=(val_array, y_val.to_numpy()),
                        epochs=80, batch_size=32, verbose=1,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                                    restore_best_weights=True)])
    probability = model.predict(test_array, verbose=0).ravel()
    prediction = (probability >= 0.5).astype(int)
    baseline = int(y_train.mean() >= 0.5)
    metrics = {
        'train_count': len(X_train), 'validation_count': len(X_val), 'test_count': len(X_test),
        'accuracy': float(accuracy_score(y_test, prediction)),
        'f1': float(f1_score(y_test, prediction, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, probability)),
        'majority_baseline_accuracy': float(accuracy_score(y_test, np.full(len(y_test), baseline))),
        'classification_report': classification_report(y_test, prediction, output_dict=True, zero_division=0),
        'logistic_baseline_accuracy': float(accuracy_score(y_test, linear_preds)),
        'logistic_baseline_f1': float(f1_score(y_test, linear_preds, zero_division=0)),
        'logistic_baseline_roc_auc': float(roc_auc_score(y_test, linear_probs)),
        'feature_count': int(train_array.shape[1]),
    }
    (ROOT / 'models').mkdir(exist_ok=True)
    (ROOT / 'outputs/figures').mkdir(parents=True, exist_ok=True)
    model.save(ROOT / 'models/youtube_mlp.keras')
    joblib.dump(preprocessing, ROOT / 'models/preprocessor.joblib')
    joblib.dump(linear, ROOT / 'models/logistic_baseline.joblib')
    (ROOT / 'outputs/metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Treino')
    ax.plot(history.history['val_loss'], label='Validação')
    ax.set(xlabel='Época', ylabel='Loss', title='Curva de aprendizado')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'outputs/figures/loss.png', dpi=160)
    plt.close(fig)
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=['Abaixo', 'Acima'])
    plt.tight_layout()
    plt.savefig(ROOT / 'outputs/figures/confusion_matrix.png', dpi=160)
    plt.close()
    print(json.dumps({k:v for k,v in metrics.items() if k != 'classification_report'}, indent=2))
    print('Modelo, pré-processador, métricas e gráficos salvos.')


if __name__ == '__main__':
    train()
