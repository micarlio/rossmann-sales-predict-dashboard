# Projeto de Previsão de Vendas - Rossmann Stores

## Sobre o Projeto
Este projeto analisa dados de vendas da rede de farmácias Rossmann e desenvolve um modelo preditivo para prever as vendas futuras das lojas. O projeto inclui análise exploratória de dados, tratamento de dados, engenharia de features, modelagem preditiva e um dashboard interativo para visualização dos resultados.

## Estrutura do Projeto

### Dataset
- **Dados brutos**: Contém os arquivos originais `store.csv` e `train.csv`
- **Dados processados**: Armazena os dataframes após o tratamento e processamento

### Notebooks
- **01 - EDA e Tratamento das Vendas**: Análise exploratória e limpeza dos dados de vendas
- **02 - EDA e Tratamento das Lojas**: Análise exploratória e limpeza dos dados das lojas
- **03 - União dos Dataframes processados**: Combinação dos dados tratados
- **04 - Engenharia de features**: Criação de novas variáveis para o modelo
- **05 - Modelagem Preditiva e Avaliação**: Desenvolvimento e avaliação do modelo de previsão

### Dashboard
Dashboard interativo desenvolvido com Dash que apresenta:
- Contextualização do problema
- Processo de limpeza de dados
- Análise preliminar dos dados
- Dashboard geral com KPIs e visualizações
- Análise específica de lojas
- Visualizações 3D
- Previsão de vendas

## Como executar o projeto

### Requisitos
```
pip install -r dashboard/requirements.txt
```

### Executar o Dashboard
```
python dashboard/app.py
```

## Tecnologias Utilizadas
- Python
- Pandas
- NumPy
- Scikit-learn
- Dash
- Plotly
- Bootstrap

## Autor
Micarlo Teixeira 