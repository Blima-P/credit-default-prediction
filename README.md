# Previsao de inadimplencia de credito

Projeto de aprendizado para portfolio: explorar dados tabulares, construir
modelos de Machine Learning e comparar com uma rede neural de Deep Learning.
Esta primeira etapa apenas carrega e apresenta os dados; nenhum modelo foi
treinado ainda.

## Problema

Prever se um cliente ficara inadimplente no proximo mes usando informacoes
disponiveis sobre credito, pagamentos e faturas anteriores.

- Dataset: [Default of Credit Card Clients, UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
- Fonte: Yeh, I. (2009), DOI: https://doi.org/10.24432/C55S3H.
- Licenca indicada pela UCI: CC BY 4.0.
- 30.000 clientes de Taiwan, com dados de 2005.
- 23 variaveis de entrada; `ID` e um identificador, nao uma feature.
- Alvo: `default payment next month` (1 = inadimplente, 0 = nao inadimplente).

Os dados sao historicos e de uma populacao especifica. Este projeto e
educacional, nao uma ferramenta para tomar decisoes reais de credito.

## Executar no Windows (PowerShell)

Requer Python 3.13. Na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\explore_data.py
```

O primeiro uso baixa o arquivo publico da UCI para `data\raw`.
Os proximos usos reutilizam o arquivo local. Os dados e o ambiente virtual
nao devem ser enviados ao GitHub.

## Percurso de aprendizado

1. Entender o alvo, as colunas e o desequilibrio entre as classes.
2. Separar treino e teste com estratificacao antes de ajustar transformacoes.
3. Explorar o conjunto de treino e construir pipelines com scikit-learn.
4. Comparar um baseline simples, regressao logistica e random forest.
5. Avaliar precision, recall, PR-AUC, ROC-AUC e matriz de confusao.
6. Adicionar uma rede neural e comparar sob a mesma separacao de dados.

O conjunto de teste deve ficar reservado para a avaliacao final. Ajustes
de modelos e de limiar devem usar apenas treino e validacao.

## Primeira atividade

Execute o script e responda:

1. O que o modelo tentara prever?
2. Por que remover `ID` das entradas?
3. Se sempre prevermos a classe mais comum, qual sera a acuracia?
4. Por que essa acuracia, sozinha, pode ser enganosa?

Antes de avancar para modelos, entenda essas respostas e consulte a
documentacao das variaveis na pagina da UCI.
