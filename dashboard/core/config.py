VERMELHO_ROSSMANN = '#E3001B'
AZUL_ESCURO = '#002346'
CINZA_NEUTRO = '#5E676F'
FUNDO_CINZA_CLARO = '#F5F7FA'
BRANCO_NEUTRO = '#FFFFFF'
VERDE_DESTAQUE = '#27AE60'
AMARELO_DESTAQUE = '#F2C94C'
AZUL_DESTAQUE = '#2D9CDB'

PALETA_CORES_GRAFICO = ['#002346', '#E3001B', '#2D9CDB', '#27AE60', '#F2C94C', '#9B59B6', '#34495E']
TEMPLATE_PLOTLY = 'rossmann_template'

MAPEAMENTO_DIAS_SEMANA = {1: 'Segunda', 2: 'Terça', 3: 'Quarta', 4: 'Quinta', 5: 'Sexta', 6: 'Sábado', 7: 'Domingo'}
ORDEM_DIAS_SEMANA = list(MAPEAMENTO_DIAS_SEMANA.values())

ALTURA_GRAFICO, ALTURA_GRAFICO_LARGURA_TOTAL = 450, 550

COLUNAS_NUMERICAS_VENDAS = ['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday']
COLUNAS_NUMERICAS_LOJAS_PARA_PLOTAR = ['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']

import plotly.graph_objects as go
import plotly.io as pio

TEMPLATE_ROSSMANN = go.layout.Template(
    layout=go.Layout(
        colorway=PALETA_CORES_GRAFICO,
        font=dict(family='Poppins', color=AZUL_ESCURO),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor='white'),
        title=dict(x=0.05),
        margin=dict(l=40, r=20, t=50, b=40)
    )
)

pio.templates[TEMPLATE_PLOTLY] = TEMPLATE_ROSSMANN
pio.templates.default = TEMPLATE_PLOTLY

DESCRICOES_COLUNAS = {
    'Store': 'Identificador único de cada loja Rossmann',
    'DayOfWeek': 'Dia da semana (1 = Segunda-feira, ..., 7 = Domingo)',
    'Date': 'Data do registro',
    'Sales': 'Valor das vendas no dia',
    'Customers': 'Número de clientes no dia',
    'Open': 'Indicador se a loja estava aberta (0 = fechada, 1 = aberta)',
    'Promo': 'Indica se a loja está realizando uma promoção no dia',
    'StateHoliday': 'Indica feriado estadual. a = feriado público, b = Páscoa, c = Natal, 0 = Nenhum',
    'SchoolHoliday': 'Indicador se a loja foi afetada pelo fechamento de escolas públicas',
    'StoreType': 'Tipo da loja (a, b, c, d)',
    'Assortment': 'Nível de sortimento: a = básico, b = extra, c = estendido',
    'CompetitionDistance': 'Distância em metros até o competidor mais próximo',
    'CompetitionOpenSinceMonth': 'Mês em que o competidor mais próximo abriu',
    'CompetitionOpenSinceYear': 'Ano em que o competidor mais próximo abriu',
    'Promo2': 'Promoção contínua e consecutiva (0 = loja não está participando, 1 = loja participando)',
    'Promo2SinceWeek': 'Semana do calendário em que a loja começou a participar da Promo2',
    'Promo2SinceYear': 'Ano em que a loja começou a participar da Promo2',
    'PromoInterval': 'Intervalos consecutivos em que Promo2 é iniciada, nomeando os meses em que a promoção é reiniciada'
} 