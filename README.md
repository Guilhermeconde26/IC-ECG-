🔧 Projeto de ECG com Arduino e Classificação de Arritmias

Este projeto combina aquisição de sinal cardíaco via Arduino com processamento e classificação de arritmias utilizando um modelo de Random Forest. O objetivo é desenvolver um sistema simples e acessível para detecção automática de padrões cardíacos anormais a partir de sinais ECG.

📡 Aquisição do Sinal

Captura do traçado ECG em tempo real usando Arduino.

Pré-processamento do sinal (filtros, normalização e remoção de ruídos).

Extração de características relevantes para análise.

🤖 Classificação de Arritmias

Treinamento de um classificador Random Forest usando dados do MIT-BIH Arrhythmia Database.

Identificação das principais classes de batimentos (N, V, A, L, entre outras).

Foco em precisão e interpretabilidade.

🎯 Objetivo

Criar uma pipeline completa — da leitura do sinal ao diagnóstico automatizado — para apoiar estudos, prototipagem e aplicações educacionais em engenharia biomédica e ciência de dados.

🚀 Tecnologias utilizadas

Arduino (aquisição do sinal)

Python (NumPy, SciPy, scikit-learn)

Random Forest Classifier

MIT-BIH Arrhythmia Database
