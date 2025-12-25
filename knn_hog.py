import os
import cv2
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
import time

start_time = time.time()

# Mapeamento dos prefixos
prefix_to_label = {
    "md": "Maio", "jd": "Junho", "jt": "Julho", "ad": "Agosto",
    "j": "Janeiro", "f": "Fevereiro", "m": "Março", "a": "Abril",
    "s": "Setembro", "o": "Outubro", "n": "Novembro", "d": "Dezembro"
}

def detect_label_from_filename(filename):
    for prefix in sorted(prefix_to_label.keys(), key=lambda x: -len(x)):
        if filename.startswith(prefix):
            return prefix_to_label[prefix]
    return None

def preprocess_image(img_gray):
    return cv2.equalizeHist(img_gray)

def extract_hog_features(image):
    features = hog(image,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')
    return features

def carregar_base_hog(diretorio):
    X, y = [], []
    for file in os.listdir(diretorio):
        if file.lower().endswith(".bmp"):
            label = detect_label_from_filename(file.lower())
            if label:
                path = os.path.join(diretorio, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (128, 128))
                img = preprocess_image(img)
                features = extract_hog_features(img)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Carrega base
X, y = carregar_base_hog("Base de Dados - Meses do Ano")
print("\nBase carregada:", X.shape)

# Codifica rótulos
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Divide treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=100)),
    ("clf", KNeighborsClassifier())
])

# Hiperparâmetros para busca
param_grid = {
    "clf__n_neighbors": [3, 5, 7, 9],
    "clf__metric": ["euclidean", "manhattan"]
}

# Validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Busca em grade com validação cruzada
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Predição no conjunto de teste
y_pred = grid.predict(X_test)
y_test_decoded = encoder.inverse_transform(y_test)
y_pred_decoded = encoder.inverse_transform(y_pred)

# Relatório e matriz de confusão
ordered_labels = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                  'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

print("\nMelhores parâmetros encontrados:")
print(grid.best_params_)

print("\nRelatório de Classificação (conjunto de teste):")
print(classification_report(y_test_decoded, y_pred_decoded, labels=ordered_labels))

cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=ordered_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=ordered_labels, yticklabels=ordered_labels)
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Tempo total
end_time = time.time()
elapsed = end_time - start_time
print(f"\nTempo total de execução: {elapsed:.2f} segundos ({elapsed / 60:.2f} minutos)")
