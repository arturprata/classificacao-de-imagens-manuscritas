from pathlib import Path
import numpy as np
import cv2
import os
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from lbp import lbp
import time

start_time = time.time()

# Mapeamento prefixo
prefix_to_label = {
    "md": "Maio", "jd": "Junho", "jt": "Julho", "ad": "Agosto",
    "j": "Janeiro", "f": "Fevereiro", "m": "Março", "a": "Abril",
    "s": "Setembro", "o": "Outubro", "n": "Novembro", "d": "Dezembro"
}

# Ordem cronológica
ordem_meses = [
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
]

def detect_label_from_filename(filename):
    for prefix in sorted(prefix_to_label.keys(), key=lambda x: -len(x)):
        if filename.startswith(prefix):
            return prefix_to_label[prefix]
    return None

def preprocess_image(img_gray):
    return cv2.equalizeHist(img_gray)

def carregar_base_lbp(diretorio):
    X, y = [], []
    for file in os.listdir(diretorio):
        if file.lower().endswith(".bmp"):
            label = detect_label_from_filename(file.lower())
            if label:
                path = os.path.join(diretorio, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (96, 96))
                img = preprocess_image(img)
                features = lbp(img)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Carrega base
X, y = carregar_base_lbp("Base de Dados - Meses do Ano")
print("Base carregada:", X.shape)
print("Distribuição por classe:", Counter(y))

# Codifica rótulos
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Divide base: 80% treino / 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Define número máximo de componentes PCA dinamicamente
max_pca = min(X_train.shape[0], X_train.shape[1])
n_components_pca = min(30, max_pca)  # Usa 30 ou menos se necessário

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n_components_pca)),
    ("clf", KNeighborsClassifier())
])

# Hiperparâmetros
param_grid = {
    "clf__n_neighbors": [3, 5, 7, 9],
    "clf__metric": ["euclidean", "manhattan"]
}

# Validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Avaliação final no conjunto de teste
y_pred = grid.predict(X_test)
y_test_decoded = le.inverse_transform(y_test)
y_pred_decoded = le.inverse_transform(y_pred)

print("\nMelhores parâmetros encontrados:")
print(grid.best_params_)

print("\nRelatório de Classificação (conjunto de teste):")
print(classification_report(y_test_decoded, y_pred_decoded, labels=ordem_meses))

cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=ordem_meses)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=ordem_meses, yticklabels=ordem_meses)
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Tempo total
end_time = time.time()
elapsed = end_time - start_time
print(f"\nTempo total de execução: {elapsed:.2f} segundos ({elapsed / 60:.2f} minutos)")