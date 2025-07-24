# Projeto de Previsão de Vendas - Rossmann Sales

## 1. Visão Geral do Projeto

Este projeto tem como objetivo desenvolver um modelo de machine learning para prever as vendas diárias da rede de drogarias Rossmann. A capacidade de prever vendas com precisão é crucial para otimizar o gerenciamento de estoque, o planejamento de pessoal e as estratégias de marketing.

O projeto abrange todo o ciclo de vida de um projeto de ciência de dados, desde a coleta e limpeza de dados até a implantação de um dashboard interativo que permite aos stakeholders explorar os dados e as previsões do modelo.

## 2. O Problema de Negócio

A Rossmann opera milhares de drogarias em toda a Europa. A empresa busca otimizar suas operações e maximizar a lucratividade. Uma previsão de vendas precisa pode ajudar a:

*   **Gerenciamento de Estoque**: Garantir que os produtos certos estejam disponíveis na quantidade certa, no momento certo, evitando excesso de estoque e rupturas.
*   **Planejamento de Pessoal**: Alocar o número adequado de funcionários para cada loja em cada dia, garantindo um bom atendimento ao cliente sem custos excessivos.
*   **Estratégias de Marketing**: Avaliar o impacto de promoções e feriados nas vendas, permitindo um planejamento de marketing mais eficaz.

## 3. Análise Exploratória de Dados (EDA)

A análise exploratória de dados foi realizada para entender as principais características dos dados e identificar padrões e tendências. As principais descobertas incluem:

*   **Sazonalidade**: As vendas apresentam forte sazonalidade semanal, com picos no início da semana e quedas nos fins de semana. Há também uma sazonalidade anual, com picos de vendas em feriados como o Natal.
*   **Promoções**: As promoções têm um impacto significativo nas vendas, com um aumento médio de 20-30% nas vendas durante os dias de promoção.
*   **Feriados**: Feriados públicos e escolares também afetam as vendas, com algumas lojas fechando em feriados públicos.
*   **Tipos de Loja e Sortimento**: Diferentes tipos de loja (A, B, C, D) e sortimento (básico, extra, estendido) apresentam diferentes padrões de vendas.

## 4. Engenharia de Features

Para melhorar o desempenho do modelo, foram criadas as seguintes features:

*   **Features Temporais**: Dia da semana, semana do ano, mês, ano, dia do mês.
*   **Features de Competição**: Tempo desde a abertura do concorrente mais próximo.
*   **Features de Promoção**: Tempo desde o início da promoção contínua (Promo2).
*   **Ticket Médio**: Vendas por cliente, para entender o comportamento de compra.

## 5. Limpeza e Tratamento de Dados

O processo de limpeza de dados incluiu:

*   **Tratamento de Valores Ausentes**: Preenchimento de valores ausentes na distância da concorrência com o valor máximo, e em outras colunas com base em estratégias como a média ou a moda.
*   **Conversão de Tipos de Dados**: Conversão de colunas de data para o formato datetime e colunas categóricas para os tipos apropriados.
*   **Remoção de Outliers**: Identificação e tratamento de outliers em colunas como vendas e clientes.

## 6. Modelagem Preditiva

O modelo de previsão de vendas foi desenvolvido utilizando a biblioteca **Prophet**, que é especializada em modelagem de séries temporais. O Prophet é robusto a dados faltantes e outliers, e é capaz de capturar sazonalidades complexas, como as semanais e anuais, além de feriados.

O desempenho do modelo foi avaliado usando métricas como o Erro Médio Absoluto (MAE), o Erro Quadrático Médio (RMSE) e o Erro Percentual Absoluto Médio (MAPE).

## 7. Dashboard Interativo

Um dashboard interativo foi desenvolvido usando **Dash** e **Plotly** para permitir a exploração dos dados e das previsões do modelo. O dashboard inclui as seguintes seções:

*   **Visão Geral**: KPIs globais, tendências temporais e análise do impacto de promoções.
*   **Análise de Lojas**: Análise detalhada do desempenho de lojas individuais.
*   **Análise 3D**: Visualizações 3D para explorar a relação entre vendas, clientes e promoções.
*   **Previsão de Vendas**: Ferramenta para gerar previsões de vendas para lojas específicas, com diferentes horizontes de tempo e granularidades.

## 8. Tecnologias Utilizadas

*   **Linguagem de Programação**: Python
*   **Bibliotecas de Análise de Dados**: Pandas, NumPy
*   **Bibliotecas de Machine Learning**: Scikit-learn, Prophet
*   **Bibliotecas de Visualização de Dados**: Plotly, Matplotlib, Seaborn
*   **Framework de Dashboard**: Dash, Dash Bootstrap Components
*   **Servidor Web**: Gunicorn (para deploy)

## 9. Como Executar o Projeto

### Pré-requisitos

*   Python 3.8 ou superior
*   Pip (gerenciador de pacotes do Python)

### Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/rossmann-sales-predict.git
    cd rossmann-sales-predict
    ```

2.  Crie e ative um ambiente virtual (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    ```

3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

### Executando o Dashboard

Para iniciar o dashboard, execute o seguinte comando na raiz do projeto:

```bash
python dashboard/app.py
```

O dashboard estará disponível em `http://127.0.0.1:8050/`.

## 10. Estrutura do Projeto

```
rossmann-sales-predict/
├── cache-directory/      # Cache para dados e modelos
├── dashboard/            # Código-fonte do dashboard
│   ├── assets/           # Arquivos estáticos (CSS, imagens)
│   ├── callbacks/        # Callbacks do Dash
│   ├── core/             # Lógica de negócio principal
│   ├── data/             # Dados processados para o dashboard
│   ├── layouts/          # Layouts das páginas do dashboard
│   ├── __init__.py
│   └── app.py            # Ponto de entrada do dashboard
├── dataset/              # Datasets originais
├── log-depuracao/        # Logs de depuração
├── notebooks/            # Jupyter notebooks para análise e modelagem
├── .gitignore
├── .python-version
├── LICENSE
├── MANIFEST.in
├── Procfile              # Para deploy no Heroku
├── README.md
├── requirements.txt      # Dependências do projeto
└── setup.py
```

## 11. Próximos Passos

*   **Otimização de Hiperparâmetros**: Realizar uma busca mais exaustiva de hiperparâmetros para o modelo Prophet.
*   **Deploy em Nuvem**: Fazer o deploy do dashboard em uma plataforma de nuvem como Heroku, AWS ou Google Cloud.
*   **Testes Automatizados**: Implementar testes unitários e de integração para garantir a qualidade do código.
*   **Monitoramento do Modelo**: Implementar um sistema para monitorar o desempenho do modelo em produção e retreiná-lo quando necessário.

## Autor

Micarlo Teixeira