# Comitê de classificadores

- [Comitê de classificadores](#comitê-de-classificadores)
- [Equipe](#equipe)
- [Base de dados](#base-de-dados)
  - [Colunas da tabela](#colunas-da-tabela)
- [Algoritmos](#algoritmos)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Redes Bayesianas](#redes-bayesianas)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [Random Forest](#random-forest)
  - [Multilayer Perceptron (MLP)](#multilayer-perceptron-mlp)
- [Execução](#execução)
  - [Métricas](#métricas)
    - [Acurácia (Accuracy)](#acurácia-accuracy)
    - [Precisão (Precision)](#precisão-precision)
    - [Revocação (Recall)](#revocação-recall)
    - [Suporte (Support)](#suporte-support)
    - [Macro Average (Média Macro)](#macro-average-média-macro)
    - [Weighted Average (Média Ponderada)](#weighted-average-média-ponderada)

# Equipe

* Cristian Prochnow
* Gustavo Henrique Dias
* Lucas Willian de Souza Serpa
* Marlon de Souza
* Ryan Gabriel Mazzei Bromati

# Base de dados

Para o desenvolvimento do projeto, usaremos a base [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), que se refere a uma pesquisa feita com mais de 400 mil norte-americanos que tinham algum fator suscetível ao início de diabete ou então apresentavam hábitos que poderiam contribuir com isso.

A pesquisa foi conduzida desde 1984, e todo ano os dados eram obtidos por meio de questionamento direto para com os pacientes, juntamente com análise estatística dos tratamentos feitos e outros processos relacionados. A base de dados que foi disponibilizada nessa postagem do Kaggle é relacionada aos dados relacionados ao ano de 2015, e é dividida em 3 bases.

A primeira delas é a que usaremos, ao qual possui 253.680 registros e são os dados das conversas tidas diretamente com os pacientes que fizeram parte do processo. Nessa base temos então 21 *features* disponíveis, que resultam em 3 classes totais, equivalentes a paciente diagnosticado com diabete, com pré-diabete e que não possui.

## Colunas da tabela

Esse *dataset* possui 21 *features* para uso. Dentre elas estão algumas que indicam valores de decisão — se consumia frutas todos os dias, por exemplo — enquanto outras são mais relacionadas a agregação de dados — anotando o índice de massa corporal do paciente consultado, por exemplo.

| Coluna | Tipo de valor | Descrição |
| :---- | :---- | :---- |
| Diabetes\_012 | Decisão | Determina o diagnóstico do paciente. Então 0 é sem diabetes; 1 é pré-diabético e 2 é diabético. |
| HighBP | Decisão | Paciente possui ou não pressão alta. 0 é sem pressão alta e 1 possui pressão quadro de pressão alta. |
| HighChol | Decisão | Paciente possui ou não alto nível de colesterol no sangue. 0 é colesterol normal e 1 é colesterol alto. |
| CholCheck | Decisão | Se paciente realizou ou não exame de nível de colesterol no sangue nos últimos 5 anos. 0 é que não realizou e 1 para quem realizou. |
| BMI | Medida | Medida de Índice de Massa Corporal do paciente. |
| Smoker | Decisão | Se paciente já fumou pelo menos 5 carteiras de cigarro (100 cigarros) ao longo da vida. 0 para não e 1 para sim. |
| Stroke | Decisão | Se paciente já sofreu AVC. 0 para não e 1 para sim. |
| HeartDiseaseorAttack | Decisão | Se paciente já sofreu algum evento grave relacionado à doenças cardíacas. 0 para não e 1 para sim. |
| PhysActivity | Decisão | Se paciente praticou atividade física nos últimos 30 dias. 0 para não e 1 para sim. |
| Fruits | Decisão | Se paciente consome 1 ou mais frutas por dia. 0 para não e 1 para sim. |
| Veggies | Decisão | Se paciente consome 1 ou mais vegetais por dia. 0 para não e 1 para sim. |
| HvyAlcoholConsump | Decisão | Paciente consome bebida alcóolica em demasia (homens adultos com mais de 14 bebidas por semana e mulheres com mais de 7 por semana). 0 para não e 1 para sim. |
| AnyHealthcare | Decisão | Paciente possui algum tipo de plano ou assistência médica. 0 para não e 1 para sim. |
| NoDocbcCost | Decisão | Em algum momento nos 12 meses paciente precisou ir ao médico, mas não foi devido ao custo. 0 para não e 1 para sim. |
| GenHlth | Opções | Em uma escala de 1 a 5, sua saúde está em quanto? (1 excelente; 2 muito boa; 3 boa; 4 justa e 5 ruim) |
| MentHlth | Opções | Considerando fatores relacionados à saúde mental (estresse, depressão e problemas emocionais), por quantos dias nos últimos 30 dias paciente sentiu que saúde mental não estava boa? (escala de 1 a 30 dias) |
| PhysHlth | Opções | Considerando fatores relacionados à saúde física (lesões, dores ou desconfortos), por quantos dias nos últimos 30 dias paciente sentiu que saúde física não estava boa? (escala de 1 a 30 dias) |
| DiffWalk | Decisão | Paciente tem dificuldade ao andar ou subir escadas? 0 para não e 1 para sim. |
| Sex | Decisão | Sexo do paciente. 0 para feminino e 1 para masculino. |
| Age | Opções | Faixa etária. 1 para 18-24 anos; 9 para 60-64 e 13 para a partir de 80 anos. |
| Education | Opções | Escolaridade em uma escala de 1 a 6\. 1 para nunca frequentou a escola ou apenas educação infantil; 2 para ensino Fundamental (anos iniciais e finais) incompleto ou completo; 3 para Ensino Médio incompleto; 4 para ensino Médio completo ou equivalente; 5 para ensino Superior/Técnico incompleto e 6 para ensino Superior completo. |
| Income | Opções | Escala de 1-8 para salário anual do paciente. 1 para menos que $10.000 anuais; 5 para menos que $35.000 e 8 para $75.000 ou mais. |

# Algoritmos

Escolhemos então 5 algoritmos, para que fossem divididos entre cada integrante da equipe. Os escolhidos foram: **K-Nearest Neighbors**, **Redes Bayesianas**, **Support Vector Machine**, **Random Forest** e **Multi Layer Perceptron**.

## K-Nearest Neighbors (KNN)

O K-Nearest Neighbors (KNN) é um algoritmo de aprendizado supervisionado simples e intuitivo, utilizado para classificação e regressão. Ele funciona identificando os K vizinhos mais próximos de um ponto de dados e, com base na distância entre eles, classifica ou prevê o valor do ponto de interesse, sendo que, para classificação, a classe mais comum entre os vizinhos é escolhida. Embora seja fácil de entender e aplicar, o KNN pode ser computacionalmente caro em grandes bases de dados e sensível a dados ruidosos, além de exigir cuidados com a escolha do valor de K e a escala dos dados.

```python
# src/models/knn.py

def train_knn_classifier(X_train, y_train, X_val=None, y_val=None, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print("Validação - Acurácia:", acc)
        print("Relatório de Classificação:\n", classification_report(y_val, y_pred))
    return model
```

## Redes Bayesianas

Uma rede bayesiana é um modelo probabilístico que representa um conjunto de variáveis e suas dependências condicionais por meio de um grafo direcionado acíclico. Cada nó no grafo representa uma variável, e as arestas indicam relações de dependência entre essas variáveis. O modelo é baseado no Teorema de Bayes, que permite calcular a probabilidade de uma variável dada as outras, facilitando inferências sobre eventos desconhecidos. A equação do Teorema de Bayes é dada por:

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

```python
# src/models/bayesian_network.py

def train_bayesian_network(X_train, y_train, X_val=None, y_val=None):
    model = GaussianNB() # Rede bayesiana simples
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print("Validação - Acurácia:", acc)
        print("Relatório de Classificação:\n", classification_report(y_val, y_pred))
    return model
```

## Support Vector Machine (SVM)

O SVM (Support Vector Machine) é um algoritmo de aprendizado supervisionado utilizado principalmente para classificação e regressão. O objetivo do SVM é encontrar o hiperplano de separação que melhor divide as classes em um espaço de alta dimensão, maximizando a margem entre os pontos mais próximos de cada classe, chamados de vetores de suporte. Em casos não linearmente separáveis, o SVM pode utilizar o truque do kernel para projetar os dados em uma dimensão superior, tornando-os separáveis. O SVM é eficaz em problemas com grandes dimensões e é conhecido por sua robustez a overfitting, especialmente quando a margem de separação é grande. Ele também é bastante utilizado em problemas como detecção de padrões, análise de texto e biomedicina.

```python
# src/models/svm.py

def train_svm_classifier(X_train, y_train, X_val=None, y_val=None, C=1.0):
    model = make_pipeline(StandardScaler(), LinearSVC(dual=False, max_iter=10000, C=C))
    
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print("Validação - Acurácia:", acc)
        print("Relatório de Classificação:\n", classification_report(y_val, y_pred))

    return model
```

## Random Forest

O Random Forest é um algoritmo de aprendizado supervisionado baseado em um conjunto de árvores de decisão. Ele constrói múltiplas árvores de decisão durante o treinamento e as combina para melhorar a precisão e reduzir o overfitting. A ideia principal é usar a técnica de bagging (Bootstrap Aggregating), onde cada árvore é treinada com um subconjunto aleatório dos dados, com reposição. Além disso, a cada divisão de um nó, uma seleção aleatória de características é considerada, o que aumenta a diversidade entre as árvores. No momento da previsão, o Random Forest faz a média (no caso de regressão) ou a votação (no caso de classificação) dos resultados das árvores individuais, fornecendo uma predição mais estável e precisa. O algoritmo é amplamente utilizado em tarefas como classificação, detecção de anomalias e análise de dados complexos devido à sua alta performance e capacidade de lidar com grandes volumes de dados e variáveis.

```python
# src/models/random_forest.py

def train_random_forest(x_train, y_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    rf = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(x_train_scaled, y_train)

    class Model:
        def __init__(self, scaler, classifier):
            self.scaler = scaler
            self.classifier = classifier

        def predict(self, x):
            x_scaled = self.scaler.transform(x)
            return self.classifier.predict(x_scaled)

    return Model(scaler, rf)
```

## Multilayer Perceptron (MLP)

A Multilayer Perceptron (MLP) é um tipo de rede neural artificial composta por múltiplas camadas de neurônios interconectados, organizados em camadas de entrada, ocultas e saída. Cada neurônio em uma camada recebe entradas ponderadas, aplica uma função de ativação (geralmente não linear, como ReLU ou Sigmoid) e transmite o resultado para os neurônios na próxima camada. O MLP é um modelo supervisionado utilizado para classificação e regressão, sendo treinado por retropropagação (backpropagation), onde os erros da saída são propagados de volta para ajustar os pesos da rede. O modelo é eficaz para aprender padrões complexos e não lineares, sendo amplamente utilizado em tarefas como reconhecimento de voz, visão computacional e processamento de linguagem natural. Apesar de sua capacidade de modelar relações complexas, o MLP pode ser suscetível a overfitting se não for bem regularizado.

```python
# src/models/neural_web.py

def train_neural_network(x_train, y_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    mlp.fit(x_train_scaled, y_train)

    class Model:
        def __init__(self, scaler, classifier):
            self.scaler = scaler
            self.classifier = classifier

        def predict(self, x):
            x_scaled = self.scaler.transform(x)
            return self.classifier.predict(x_scaled)

    return Model(scaler, mlp)
```

# Execução

A estrutura de arquivos do projeto é formada pelo arquivo de *output*, ao qual conterá os *logs* do processamento e que pode ser encontrado em `utils/app.log`.

Junto a isso, temos então a pasta `src`, onde toda a mágica acontece. Nessa pasta há o arquivo `src/main.py`, que é responsável pelo *bootstrap* do projeto e também chamar todos os **modelos** (`src/models`) de algoritmo que foram feitos. Já, o arquivo `src/data_process.py` possui funções que são primordiais para o processamento dos dados e que são chamadas também no `main.py` para auxiliar na lógica.

E, então, na pasta `src/models` temos todos os algoritmos que foram citados acima, em seus respectivos arquivos. Esses arquivos são então chamados no `main.py` para que a execução ocorra por completo, registrando os *logs*.

Para rodar o projeto, basta então executar os comandos abaixo:

```bash
$ pip install -r requirements.txt
$ python3 src/main.py
```

Após a execução, o resultado no arquivo `utils/app.log` vai ser uma estrutura como abaixo.

```bash
2025-03-27 21:19:29,883 - INFO - Data balance: Diabetes_binary
0    0.860667
1    0.139333
Name: proportion, dtype: float64
2025-03-27 21:19:29,893 - INFO - ==========Starting process id: b8aae0c3-3567-4d51-8594-f970c0e1cc87 ================
2025-03-27 21:19:30,290 - INFO - === Training and rating model : Bayesian ===
2025-03-27 21:19:31,529 - INFO - Fold 1 - Accuracy: 0.7721
2025-03-27 21:19:31,541 - INFO -               precision    recall  f1-score   support

           0       0.92      0.81      0.86     72905
           1       0.32      0.57      0.41     11655

    accuracy                           0.77     84560
   macro avg       0.62      0.69      0.63     84560
weighted avg       0.84      0.77      0.80     84560

2025-03-27 21:19:32,736 - INFO - Fold 2 - Accuracy: 0.7760
2025-03-27 21:19:32,750 - INFO -               precision    recall  f1-score   support

           0       0.92      0.81      0.86     72653
           1       0.33      0.58      0.42     11907

    accuracy                           0.78     84560
   macro avg       0.63      0.69      0.64     84560
weighted avg       0.84      0.78      0.80     84560

2025-03-27 21:19:33,970 - INFO - Fold 3 - Accuracy: 0.7723
2025-03-27 21:19:33,981 - INFO -               precision    recall  f1-score   support

           0       0.92      0.81      0.86     72776
           1       0.32      0.56      0.41     11784

    accuracy                           0.77     84560
   macro avg       0.62      0.69      0.63     84560
weighted avg       0.84      0.77      0.80     84560

2025-03-27 21:19:33,981 - INFO - === Training and rating model : Neural Network ===
2025-03-27 21:19:41,464 - INFO - Fold 1 - Accuracy: 0.8671
2025-03-27 21:19:41,479 - INFO -               precision    recall  f1-score   support

           0       0.88      0.99      0.93     72905
           1       0.59      0.12      0.20     11655

    accuracy                           0.87     84560
   macro avg       0.73      0.55      0.56     84560
weighted avg       0.84      0.87      0.83     84560

2025-03-27 21:19:55,374 - INFO - Fold 2 - Accuracy: 0.8649
2025-03-27 21:19:55,387 - INFO -               precision    recall  f1-score   support

           0       0.88      0.98      0.93     72653
           1       0.58      0.15      0.23     11907

    accuracy                           0.86     84560
   macro avg       0.73      0.56      0.58     84560
weighted avg       0.83      0.86      0.83     84560

2025-03-27 21:20:04,541 - INFO - Fold 3 - Accuracy: 0.8670
2025-03-27 21:20:04,561 - INFO -               precision    recall  f1-score   support

           0       0.88      0.99      0.93     72776
           1       0.60      0.14      0.22     11784

    accuracy                           0.87     84560
   macro avg       0.74      0.56      0.57     84560
weighted avg       0.84      0.87      0.83     84560

2025-03-27 21:20:04,561 - INFO - === Training and rating model : Random Forest ===
2025-03-27 21:20:08,289 - INFO - Fold 1 - Accuracy: 0.8598
2025-03-27 21:20:08,301 - INFO -               precision    recall  f1-score   support

           0       0.88      0.97      0.92     72905
           1       0.48      0.17      0.25     11655

    accuracy                           0.86     84560
   macro avg       0.68      0.57      0.59     84560
weighted avg       0.82      0.86      0.83     84560

2025-03-27 21:20:12,175 - INFO - Fold 2 - Accuracy: 0.8597
2025-03-27 21:20:12,185 - INFO -               precision    recall  f1-score   support

           0       0.88      0.97      0.92     72653
           1       0.51      0.17      0.25     11907

    accuracy                           0.86     84560
   macro avg       0.69      0.57      0.59     84560
weighted avg       0.82      0.86      0.83     84560

2025-03-27 21:20:15,802 - INFO - Fold 3 - Accuracy: 0.8599
2025-03-27 21:20:15,813 - INFO -               precision    recall  f1-score   support

           0       0.88      0.97      0.92     72776
           1       0.49      0.18      0.26     11784

    accuracy                           0.86     84560
   macro avg       0.69      0.57      0.59     84560
weighted avg       0.83      0.86      0.83     84560

2025-03-27 21:20:15,813 - INFO - === Training and rating model : KNN ===
2025-03-27 21:23:44,962 - INFO - Fold 1 - Accuracy: 0.8485
2025-03-27 21:23:44,975 - INFO -               precision    recall  f1-score   support

           0       0.88      0.95      0.92     72905
           1       0.40      0.19      0.26     11655

    accuracy                           0.85     84560
   macro avg       0.64      0.57      0.59     84560
weighted avg       0.81      0.85      0.83     84560

2025-03-27 21:27:13,560 - INFO - Fold 2 - Accuracy: 0.8482
2025-03-27 21:27:13,570 - INFO -               precision    recall  f1-score   support

           0       0.88      0.96      0.92     72653
           1       0.42      0.20      0.27     11907

    accuracy                           0.85     84560
   macro avg       0.65      0.58      0.59     84560
weighted avg       0.81      0.85      0.82     84560

2025-03-27 21:30:44,768 - INFO - Fold 3 - Accuracy: 0.8463
2025-03-27 21:30:44,778 - INFO -               precision    recall  f1-score   support

           0       0.88      0.95      0.91     72776
           1       0.40      0.20      0.26     11784

    accuracy                           0.85     84560
   macro avg       0.64      0.57      0.59     84560
weighted avg       0.81      0.85      0.82     84560

2025-03-27 21:30:44,778 - INFO - === Training and rating model : SVM ===
2025-03-27 21:30:46,264 - INFO - Fold 1 - Accuracy: 0.8648
2025-03-27 21:30:46,280 - INFO -               precision    recall  f1-score   support

           0       0.87      0.99      0.93     72905
           1       0.58      0.07      0.12     11655

    accuracy                           0.86     84560
   macro avg       0.72      0.53      0.53     84560
weighted avg       0.83      0.86      0.82     84560

2025-03-27 21:30:47,794 - INFO - Fold 2 - Accuracy: 0.8614
2025-03-27 21:30:47,807 - INFO -               precision    recall  f1-score   support

           0       0.87      0.99      0.92     72653
           1       0.57      0.06      0.12     11907

    accuracy                           0.86     84560
   macro avg       0.72      0.53      0.52     84560
weighted avg       0.82      0.86      0.81     84560

2025-03-27 21:30:49,344 - INFO - Fold 3 - Accuracy: 0.8640
2025-03-27 21:30:49,357 - INFO -               precision    recall  f1-score   support

           0       0.87      0.99      0.93     72776
           1       0.60      0.07      0.13     11784

    accuracy                           0.86     84560
   macro avg       0.74      0.53      0.53     84560
weighted avg       0.83      0.86      0.81     84560

2025-03-27 21:30:49,358 - INFO - === Accuracy average results ===
2025-03-27 21:30:49,358 - INFO - Bayesian: Average = 0.7735, Standard Variation = 0.0018
2025-03-27 21:30:49,358 - INFO - Neural Network: Average = 0.8663, Standard Variation = 0.0010
2025-03-27 21:30:49,358 - INFO - Random Forest: Average = 0.8598, Standard Variation = 0.0001
2025-03-27 21:30:49,358 - INFO - KNN: Average = 0.8477, Standard Variation = 0.0010
2025-03-27 21:30:49,358 - INFO - SVM: Average = 0.8634, Standard Variation = 0.0014
```

## Métricas

Esse *log* oferece a visualização das seguintes métricas:

### Acurácia (Accuracy)

A acurácia é a proporção de previsões corretas (verdadeiros positivos e verdadeiros negativos) em relação ao número total de amostras. É uma métrica simples, mas pode ser tendenciosa se o conjunto de dados for desbalanceado.

$$
\text{Acurácia} = \frac{\text{Verdadeiros Positivos} + \text{Verdadeiros Negativos}}{\text{Total de Amostras}}
$$

### Precisão (Precision)

A precisão é a proporção de previsões positivas corretas em relação ao total de previsões positivas feitas. Em outras palavras, diz o quanto o modelo está acertando quando prevê a classe positiva.

$$
\text{Precisão} = \frac{\text{Verdadeiros Positivos}}{\text{Verdadeiros Positivos + Falsos Positivos}}
$$

### Revocação (Recall)

A revocação (também conhecida como sensibilidade ou true positive rate) é a proporção de previsões positivas corretas em relação ao total de casos que realmente pertencem à classe positiva. Ela mede a capacidade do modelo de capturar todos os casos positivos reais.

$$
\text{Revocação} = \frac{\text{Verdadeiros Positivos}}{\text{Verdadeiros Positivos + Falsos Negativos}}
$$

### Suporte (Support)

O suporte refere-se ao número de ocorrências reais de cada classe no conjunto de dados. Por exemplo, o número de exemplos para cada classe positiva ou negativa. Não é uma métrica de desempenho, mas sim uma informação adicional para contextualizar as outras métricas.

### Macro Average (Média Macro)

A média macro calcula a média das métricas (precisão, revocação e F1-score) para cada classe individualmente e depois calcula a média desses valores. Ela trata todas as classes igualmente, independentemente do seu tamanho ou frequência no conjunto de dados.

$$
\text{Macro Average} = \frac{1}{N} \sum_{i=1}^{N} \text{Métrica}_i
$$

### Weighted Average (Média Ponderada)

A média ponderada calcula as métricas médias, mas leva em consideração o número de amostras de cada classe. Ou seja, classes com mais amostras terão mais peso na média. Isso é útil quando as classes estão desbalanceadas.

$$
\text{Weighted Average} = \sum_{i=1}^{N} \left( \frac{\text{Support}_i}{\text{Total}} \times \text{Métrica}_i \right)
$$