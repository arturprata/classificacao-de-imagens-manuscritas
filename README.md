
# Classificação de Imagens Manuscritas

## 📝 Descrição breve

Análise comparativa entre KNN e SVM para classificação de imagens manuscritas, usando LBP e HOG como descritores — incluindo métricas, matrizes de confusão e discussão dos resultados.

---

## 📖 Sobre o projeto

Este repositório contém a avaliação de dois classificadores supervisionados (KNN e SVM) aplicados à tarefa de reconhecimento de nomes de meses escritos à mão.

Para alimentar esses modelos, utilizamos dois descritores de textura:
- **LBP (Local Binary Patterns)**
- **HOG (Histogram of Oriented Gradients)**

---

## 🧠 Base de Dados

- 6000 imagens manuscritas no formato BMP
- 12 classes (meses do ano), 500 imagens por classe
- Rótulo extraído do prefixo do nome do arquivo

---

## ⚙️ Pipeline de Processamento

1. Conversão para escala de cinza
2. Redimensionamento (96x96 para LBP, 128x128 para HOG)
3. Equalização de histograma
4. Extração de características (LBP ou HOG)
5. Treinamento com **validação cruzada estratificada (5-fold)**
6. Otimização de hiperparâmetros via **Grid Search**
7. Divisão treino/teste: **80/20**

---

## 📊 Resultados Obtidos

| Modelo | Descritor | Acurácia | F1-score macro |
|--------|-----------|----------|----------------|
| KNN    | LBP       | 46%      | 0.45           |
| SVM    | LBP       | 56%      | 0.56           |
| KNN    | HOG       | 74%      | 0.75           |
| SVM    | HOG       | **86%**  | **0.86**       |

As melhores métricas foram obtidas com a combinação **SVM + HOG**.

---

## 📈 Tabelas e Matrizes de Confusão

As tabelas de classificação e matrizes de confusão completas estão disponíveis na pasta `/results`, incluindo métricas por classe (Precisão, Recall, F1-score).

---

## 💡 Conclusão

- O tipo de descritor tem grande influência no desempenho do classificador.
- KNN mostrou resultados modestos com LBP, mas melhorou com HOG.
- SVM com HOG foi a configuração mais robusta, destacando-se em todas as métricas.
- Como trabalho futuro, sugere-se explorar redes neurais convolucionais (CNNs) e fusão de múltiplos descritores.

---

## 📂 Estrutura do Repositório

```text
/
├── src/                    # Scripts de treinamento e avaliação
├── data/                   # Base de dados (referência ou instruções)
├── article/                # Artigo em PDF
└── README.md               # Este arquivo
```

---

## 🧪 Tecnologias Utilizadas

- Python 3.x
- NumPy, Pandas
- Scikit-learn
- Scikit-image
- OpenCV
- Matplotlib

