# dashboard/layouts/layout_contextualizacao.py
import dash_bootstrap_components as dbc
from dash import dcc, html
import pandas as pd
from ..data_loader import carregar_dados_brutos

from ..config import VERMELHO_ROSSMANN, AZUL_ESCURO, FUNDO_CINZA_CLARO, BRANCO_NEUTRO, AZUL_DESTAQUE, VERDE_DESTAQUE, AMARELO_DESTAQUE, PALETA_CORES_GRAFICO, CINZA_NEUTRO

def criar_layout_contextualizacao(dados):
    # Carrega datasets brutos para contextualização
    df_vendas_raw, df_lojas_raw = carregar_dados_brutos()
    if df_vendas_raw is None or df_lojas_raw is None:
        df_principal = pd.DataFrame()
        df_vendas_original = pd.DataFrame()
        df_caracteristicas_original = pd.DataFrame()
    else:
        df_principal = pd.merge(df_vendas_raw, df_lojas_raw, how='left', on='Store')
        df_vendas_original = df_vendas_raw
        df_caracteristicas_original = df_lojas_raw
    
    todas_colunas = []
    
    # Verifica se os DataFrames não estão vazios
    if not df_principal.empty:
        todas_colunas = sorted(df_principal.columns.tolist())
    
    # Calcula algumas estatísticas básicas com verificações
    if not df_principal.empty and 'Date' in df_principal.columns and len(df_principal['Date']) > 0:
        periodo_inicio = df_principal['Date'].min().strftime('%d/%m/%Y')
        periodo_fim = df_principal['Date'].max().strftime('%d/%m/%Y')
        total_lojas = df_principal['Store'].nunique()
        total_registros = len(df_principal)
    else:
        periodo_inicio = 'N/A'
        periodo_fim = 'N/A'
        total_lojas = 0
        total_registros = 0
    
    # Estilos base
    estilo_card_principal = {
        'borderRadius': '15px',
        'boxShadow': '0 6px 16px rgba(0, 0, 0, 0.1)',
        'background': 'linear-gradient(to right, #FFFFFF, #F8F9FA)',
        'border': 'none',
        'marginBottom': '25px',
        'overflow': 'hidden'
    }
    
    estilo_card_secundario = {
        'borderRadius': '12px',
        'boxShadow': '0 4px 10px rgba(0, 0, 0, 0.08)',
        'backgroundColor': BRANCO_NEUTRO,
        'border': 'none',
        'height': '100%',
        'transition': 'transform 0.3s ease',
    }
    
    estilo_linha_titulo = {
        'borderLeft': f'5px solid {VERMELHO_ROSSMANN}',
        'paddingLeft': '15px',
        'marginBottom': '20px'
    }

    estilo_indice = {
        'display': 'inline-flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'backgroundColor': VERMELHO_ROSSMANN,
        'color': 'white',
        'borderRadius': '50%',
        'width': '30px',
        'height': '30px',
        'fontWeight': 'bold',
        'marginRight': '10px'
    }
    
    # Função para criar títulos de seção com linha lateral
    def titulo_secao(texto, icone=None):
        if icone:
            return html.Div([
                html.Div([
                    html.I(className=f"fas {icone}", style={'fontSize': '24px', 'marginRight': '15px', 'color': VERMELHO_ROSSMANN}),
                    html.H3(texto, style={'color': AZUL_ESCURO, 'fontWeight': '700', 'margin': '0', 'fontSize': '1.8rem'}),
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style=estilo_linha_titulo)
        else:
            return html.Div([
                html.H3(texto, style={'color': AZUL_ESCURO, 'fontWeight': '700', 'margin': '0', 'fontSize': '1.8rem'})
            ], style=estilo_linha_titulo)
    
    # Função para criar cards de estatísticas
    def card_estatistica(titulo, valor, icone, cor):
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.I(className=f"fas {icone}", style={
                            'fontSize': '28px', 
                            'color': cor,
                            'backgroundColor': 'rgba(41, 182, 246, 0.1)',
                            'padding': '15px',
                            'borderRadius': '12px',
                            'marginRight': '15px'
                        }),
                    ]),
                    html.Div([
                        html.P(titulo, style={'fontSize': '0.9rem', 'color': CINZA_NEUTRO, 'marginBottom': '5px'}),
                        html.H4(valor, style={'fontWeight': '700', 'color': AZUL_ESCURO, 'marginBottom': '0'})
                    ])
                ], style={'display': 'flex', 'alignItems': 'center'})
            ])
        ], style={
            'border': 'none',
            'borderRadius': '12px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.05)',
            'height': '100%'
        })
    
    # Função para criar cartões de fatores
    def cartao_fator(titulo, descricao, icone, cor, itens=None):
        conteudo = [
            html.Div([
                html.I(className=f"fas {icone}", style={
                    'fontSize': '24px',
                    'color': 'white',
                'backgroundColor': cor,
                    'padding': '12px',
                    'borderRadius': '12px',
                    'marginBottom': '15px'
                })
            ], style={'textAlign': 'center'}),
            
            html.H5(titulo, style={
                'fontWeight': '600', 
                'color': AZUL_ESCURO, 
                'textAlign': 'center',
                'fontSize': '1.1rem',
                'marginBottom': '10px'
            }),
            
            html.P(descricao, style={
                'textAlign': 'center', 
                'color': CINZA_NEUTRO, 
                'fontSize': '0.9rem',
                'marginBottom': '15px'
            })
        ]
        
        if itens:
            conteudo.append(
                html.Ul([
                    html.Li(item, style={'marginBottom': '8px', 'fontSize': '0.9rem'}) 
                    for item in itens
                ], style={'paddingLeft': '20px', 'marginBottom': '0'})
            )
            
        return dbc.Card(
            dbc.CardBody(conteudo),
            style={
                **estilo_card_secundario,
                'transition': 'transform 0.2s, box-shadow 0.2s',
                'cursor': 'pointer',
                ':hover': {
                    'transform': 'translateY(-5px)',
                    'boxShadow': '0 8px 15px rgba(0,0,0,0.1)'
                }
            },
            className="h-100"
        )
    
    # Função para criar cards de dicionário
    def cartao_dicionario(titulo, descricao, dataset, cor):
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Span(dataset, style={
                        'backgroundColor': cor,
                        'color': 'white',
                        'padding': '5px 10px',
                        'borderRadius': '20px',
                        'fontSize': '0.7rem',
                        'fontWeight': '600',
                        'marginRight': '10px'
                    }),
                    html.H6(titulo, style={'fontWeight': '600', 'margin': '0', 'display': 'inline'})
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'backgroundColor': 'white', 'borderBottom': f'2px solid {cor}'}),
            dbc.CardBody([
                html.P(descricao, style={'fontSize': '0.9rem'})
            ])
        ], style={
            'borderRadius': '10px',
            'overflow': 'hidden',
            'border': 'none',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.08)',
            'marginBottom': '15px'
        })
    
    # Seção de estilos para as tabelas de amostra
    estilo_tabela = {
        'boxShadow': '0 2px 5px rgba(0,0,0,0.05)',
        'borderRadius': '8px',
        'overflowX': 'auto'
    }
    
    # Começando o layout real
    layout = dbc.Container([
        # Cabeçalho com Hero Section
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Img(
                                src='/assets/images/rossmann_logo_colorida.png',
                                style={'maxWidth': '180px', 'marginBottom': '20px'}
                            ),
                            html.H1("Análise de Vendas Rossmann", style={
                                'color': AZUL_ESCURO,
                                'fontWeight': '800',
                                'fontSize': '2.5rem',
                                'marginBottom': '15px'
                            }),
                            html.P([
                                "Dashboard analítico desenvolvido para identificar padrões e ",
                                html.Strong("prever vendas futuras"), 
                                " nas lojas da rede Rossmann, proporcionando insights estratégicos para a tomada de decisões."
                            ], style={
                                'fontSize': '1.1rem',
                                'lineHeight': '1.6',
                                'color': CINZA_NEUTRO,
                                'marginBottom': '20px'
                            }),
                            
                            html.Div([
                                dbc.Button([
                                    html.I(className="fas fa-chart-line me-2"),
                                    "Explorar Dashboard"
                                ], color="danger", href="/dashboard", size="lg", className="me-2"),
                                
                                dbc.Button([
                                    html.I(className="fas fa-store me-2"),
                                    "Análise por Loja"
                                ], color="primary", outline=True, href="/analise-lojas", size="lg"),
                                
                                html.Button(html.I(className="fas fa-expand-alt"), 
                                           id={'type': 'botao-tela-cheia', 'index': 'contextualizacao'}, 
                                           className='btn btn-outline-secondary ms-2', 
                                           title='Tela Cheia',
                                           style={'marginLeft': '10px'})
                            ], className="d-flex align-items-center"),
                            
                            dcc.Store(id={'type': 'saida-tela-cheia', 'index': 'contextualizacao'})
                        ])
                    ], xs=12, sm=12, md=7, lg=8),
                    
                    dbc.Col([
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Div([
                                            html.I(className="fas fa-calendar", style={
                                                'fontSize': '28px',
                                                'color': VERMELHO_ROSSMANN,
                                                'backgroundColor': 'rgba(220, 53, 69, 0.1)',
                                                'padding': '15px',
                                                'borderRadius': '12px',
                                                'marginBottom': '10px',
                                                'width': '60px',
                                                'height': '60px',
                                                'display': 'flex',
                                                'alignItems': 'center',
                                                'justifyContent': 'center'
                                            })
                                        ], style={'textAlign': 'center'}),
                                        html.P("Período de Análise", style={
                                            'fontSize': '1rem',
                                            'color': CINZA_NEUTRO,
                                            'marginBottom': '5px',
                                            'textAlign': 'center'
                                        }),
                                        html.H4(f"{periodo_inicio} - {periodo_fim}", style={
                                            'fontWeight': '700',
                                            'color': AZUL_ESCURO,
                                            'marginBottom': '0',
                                            'textAlign': 'center',
                                            'fontSize': '1.1rem'
                                        })
                                    ], style={
                                        'backgroundColor': BRANCO_NEUTRO,
                                        'borderRadius': '15px',
                                        'padding': '20px',
                                        'height': '100%',
                                        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                                    })
                                ], xs=12, className="mb-3"),
                                
                                dbc.Col([
                                    html.Div([
                                        html.Div([
                                            html.I(className="fas fa-store", style={
                                                'fontSize': '28px',
                                                'color': AZUL_DESTAQUE,
                                                'backgroundColor': 'rgba(0, 123, 255, 0.1)',
                                                'padding': '15px',
                                                'borderRadius': '12px',
                                                'marginBottom': '10px',
                                                'width': '60px',
                                                'height': '60px',
                                                'display': 'flex',
                                                'alignItems': 'center',
                                                'justifyContent': 'center'
                                            })
                                        ], style={'textAlign': 'center'}),
                                        html.P("Total de Lojas", style={
                                            'fontSize': '1rem',
                                            'color': CINZA_NEUTRO,
                                            'marginBottom': '5px',
                                            'textAlign': 'center'
                                        }),
                                        html.H4(f"{total_lojas:,}", style={
                                            'fontWeight': '700',
                                            'color': AZUL_ESCURO,
                                            'marginBottom': '0',
                                            'textAlign': 'center',
                                            'fontSize': '1.4rem'
                                        })
                                    ], style={
                                        'backgroundColor': BRANCO_NEUTRO,
                                        'borderRadius': '15px',
                                        'padding': '20px',
                                        'height': '100%',
                                        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                                    })
                                ], xs=12, sm=6, className="mb-3"),
                                
                                dbc.Col([
                                    html.Div([
                                        html.Div([
                                            html.I(className="fas fa-database", style={
                                                'fontSize': '28px',
                                                'color': VERDE_DESTAQUE,
                                                'backgroundColor': 'rgba(40, 167, 69, 0.1)',
                                                'padding': '15px',
                                                'borderRadius': '12px',
                                                'marginBottom': '10px',
                                                'width': '60px',
                                                'height': '60px',
                                                'display': 'flex',
                                                'alignItems': 'center',
                                                'justifyContent': 'center'
                                            })
                                        ], style={'textAlign': 'center'}),
                                        html.P("Registros", style={
                                            'fontSize': '1rem',
                                            'color': CINZA_NEUTRO,
                                            'marginBottom': '5px',
                                            'textAlign': 'center'
                                        }),
                                        html.H4(f"{total_registros:,}", style={
                                            'fontWeight': '700',
                                            'color': AZUL_ESCURO,
                                            'marginBottom': '0',
                                            'textAlign': 'center',
                                            'fontSize': '1.4rem'
                                        })
                                    ], style={
                                        'backgroundColor': BRANCO_NEUTRO,
                                        'borderRadius': '15px',
                                        'padding': '20px',
                                        'height': '100%',
                                        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                                    })
                                ], xs=12, sm=6, className="mb-3")
                            ])
                        ], style={
                            'backgroundColor': FUNDO_CINZA_CLARO,
                            'borderRadius': '15px',
                            'padding': '20px',
                            'height': '100%',
                            'boxShadow': 'inset 0 2px 10px rgba(0,0,0,0.05)'
                        })
                    ], xs=12, sm=12, md=5, lg=4, className="mt-4 mt-md-0")
                ])
            ])
        ], style=estilo_card_principal),
        
        # Seção 1: Sobre o Projeto
        dbc.Card([
            dbc.CardBody([
                titulo_secao("Sobre o Projeto", "fa-project-diagram"),
                
            dbc.Row([
                dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div([
                                        html.Span("01", style={
                                            'backgroundColor': VERMELHO_ROSSMANN,
                                            'color': 'white',
                                            'borderRadius': '50%',
                                            'width': '36px',
                                            'height': '36px',
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'center',
                                            'fontWeight': '600',
                                            'fontSize': '1rem',
                                            'marginBottom': '15px'
                                        })
                                    ]),
                                    html.H4("Contexto", style={
                                        'color': AZUL_ESCURO,
                                        'fontWeight': '600',
                                        'marginBottom': '15px',
                                        'fontSize': '1.5rem'
                                    }),
                                    html.P([
                                        "A Rossmann é uma das maiores redes de drogarias da Europa, ",
                                        html.Strong("operando mais de 3.000 lojas em 7 países. "),
                                        "Este dashboard foi desenvolvido a partir de dados reais da Rossmann, disponibilizados em uma competição ",
                                        "do Kaggle, com o objetivo de analisar fatores que impactam as vendas e criar modelos preditivos eficientes."
                                    ], style={
                                        'color': CINZA_NEUTRO,
                                        'lineHeight': '1.6',
                                        'fontSize': '1rem',
                                        'marginBottom': '0'
                                    })
                                ])
                            ])
                        ], style={
                            'height': '100%',
                            'border': 'none',
                            'borderRadius': '15px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                            'backgroundColor': BRANCO_NEUTRO,
                            'transition': 'transform 0.2s ease',
                            'cursor': 'default',
                            'hover': {
                                'transform': 'translateY(-5px)',
                                'boxShadow': '0 6px 12px rgba(0,0,0,0.15)'
                            }
                        }, className="mb-4")
                    ], xs=12, md=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div([
                                        html.Span("02", style={
                                            'backgroundColor': AZUL_DESTAQUE,
                                            'color': 'white',
                                            'borderRadius': '50%',
                                            'width': '36px',
                                            'height': '36px',
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'center',
                                            'fontWeight': '600',
                                            'fontSize': '1rem',
                                            'marginBottom': '15px'
                                        })
                                    ]),
                                    html.H4("Objetivos", style={
                                        'color': AZUL_ESCURO,
                                        'fontWeight': '600',
                                        'marginBottom': '15px',
                                        'fontSize': '1.5rem'
                                    }),
                                    html.P([
                                        "Os principais objetivos deste dashboard são ",
                                        html.Strong("identificar padrões sazonais, avaliar o impacto de promoções, "),
                                        "compreender as diferenças entre tipos de lojas e criar um modelo preditivo de vendas. ",
                                        "Estas análises são essenciais para otimizar estratégias comerciais e operacionais."
                                    ], style={
                                        'color': CINZA_NEUTRO,
                                        'lineHeight': '1.6',
                                        'fontSize': '1rem',
                                        'marginBottom': '0'
                                    })
                                ])
                            ])
                        ], style={
                            'height': '100%',
                            'border': 'none',
                            'borderRadius': '15px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                            'backgroundColor': BRANCO_NEUTRO,
                            'transition': 'transform 0.2s ease',
                            'cursor': 'default',
                            'hover': {
                                'transform': 'translateY(-5px)',
                                'boxShadow': '0 6px 12px rgba(0,0,0,0.15)'
                            }
                        }, className="mb-4")
                    ], xs=12, md=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                    html.Div([
                                        html.Span("03", style={
                                            'backgroundColor': VERDE_DESTAQUE,
                                            'color': 'white',
                                            'borderRadius': '50%',
                                            'width': '36px',
                                            'height': '36px',
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'center',
                                            'fontWeight': '600',
                                            'fontSize': '1rem',
                                            'marginBottom': '15px'
                                        })
                                    ]),
                                    html.H4("Dados Utilizados", style={
                                        'color': AZUL_ESCURO,
                                        'fontWeight': '600',
                                        'marginBottom': '15px',
                                        'fontSize': '1.5rem'
                                    }),
                                    html.P([
                                        "Os dados contêm ",
                                        html.Strong("vendas diárias de lojas Rossmann entre 2013 e 2015"),
                                        ", incluindo variáveis como promoções, feriados, competição e características específicas de lojas. ",
                                        "A análise envolveu etapas de limpeza, engenharia de atributos e modelagem estatística."
                                    ], style={
                                        'color': CINZA_NEUTRO,
                                        'lineHeight': '1.6',
                                        'fontSize': '1rem',
                                        'marginBottom': '0'
                                    })
                                ])
                            ])
                        ], style={
                            'height': '100%',
                            'border': 'none',
                            'borderRadius': '15px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                            'backgroundColor': BRANCO_NEUTRO,
                            'transition': 'transform 0.2s ease',
                            'cursor': 'default',
                            'hover': {
                                'transform': 'translateY(-5px)',
                                'boxShadow': '0 6px 12px rgba(0,0,0,0.15)'
                            }
                        }, className="mb-4")
                    ], xs=12, md=6),
                    
                            dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Div([
                                        html.Span("04", style={
                                            'backgroundColor': AMARELO_DESTAQUE,
                                            'color': 'white',
                                            'borderRadius': '50%',
                                            'width': '36px',
                                            'height': '36px',
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'center',
                                            'fontWeight': '600',
                                            'fontSize': '1rem',
                                            'marginBottom': '15px'
                                        })
                                    ]),
                                    html.H4("Metodologia", style={
                                        'color': AZUL_ESCURO,
                                        'fontWeight': '600',
                                        'marginBottom': '15px',
                                        'fontSize': '1.5rem'
                                    }),
                                    html.P([
                                        "Este projeto segue a metodologia ",
                                        html.Strong("CRISP-DM (Cross Industry Standard Process for Data Mining)"),
                                        ", com foco em entendimento do negócio, preparação de dados, visualizações interativas e modelagem. ",
                                        "A abordagem permitiu insights acionáveis para decisões estratégicas de negócio."
                                    ], style={
                                        'color': CINZA_NEUTRO,
                                        'lineHeight': '1.6',
                                        'fontSize': '1rem',
                                        'marginBottom': '0'
                                    })
                                ])
                            ])
                        ], style={
                            'height': '100%',
                            'border': 'none',
                            'borderRadius': '15px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                            'backgroundColor': BRANCO_NEUTRO,
                            'transition': 'transform 0.2s ease',
                            'cursor': 'default',
                            'hover': {
                                'transform': 'translateY(-5px)',
                                'boxShadow': '0 6px 12px rgba(0,0,0,0.15)'
                            }
                        }, className="mb-4")
                    ], xs=12, md=6)
                ])
            ])
        ], style=estilo_card_principal),
        
        # Seção 2: Fatores que Influenciam as Vendas
        dbc.Card([
            dbc.CardBody([
                titulo_secao("Fatores que Influenciam as Vendas", "fa-chart-line"),
                
                dbc.Row([
                    dbc.Col([
                        cartao_fator(
                            "Sazonalidade",
                            "Padrões temporais que afetam o comportamento das vendas", 
                            "fa-calendar-alt",
                            VERMELHO_ROSSMANN,
                            [
                                "Variação entre dias da semana (picos nos finais de semana)",
                                "Ciclos mensais associados a datas de pagamento",
                                "Padrões anuais e sazonais (feriados, eventos)"
                            ]
                        )
                    ], md=4, className="mb-4"),
                    
                    dbc.Col([
                        cartao_fator(
                            "Promoções",
                            "Estratégias promocionais e seu impacto nas vendas", 
                            "fa-tags",
                            AZUL_DESTAQUE,
                            [
                                "Promoções regulares (Promo) - aumento médio de 12% nas vendas",
                                "Promoções consecutivas (Promo2) - estratégia de longo prazo",
                                "Intervalos promocionais específicos durante o ano"
                            ]
                        )
                    ], md=4, className="mb-4"),
                    
                    dbc.Col([
                        cartao_fator(
                            "Características da Loja",
                            "Atributos específicos de cada unidade", 
                            "fa-store",
                            VERDE_DESTAQUE,
                            [
                                "Tipologia (a, b, c, d) - diferentes formatos de operação",
                                "Sortimento de produtos (básico, extra, estendido)",
                                "Localização e proximidade de concorrentes"
                            ]
                        )
                    ], md=4, className="mb-4")
                ]),
                
                dbc.Row([
                    dbc.Col([
                        cartao_fator(
                            "Feriados",
                            "Impacto de datas comemorativas nas operações", 
                            "fa-plane",
                            AMARELO_DESTAQUE,
                            [
                                "Feriados estaduais (públicos, Páscoa, Natal)",
                                "Férias escolares e seu efeito nas vendas",
                                "Padrão de fechamento de lojas em feriados específicos"
                            ]
                        )
                    ], md=6, className="mb-4"),
                    
                    dbc.Col([
                        cartao_fator(
                            "Fatores Externos",
                            "Elementos contextuais que afetam o desempenho", 
                            "fa-users",
                            PALETA_CORES_GRAFICO[5],
                            [
                                "Comportamento e perfil dos clientes por região",
                                "Condições econômicas e indicadores sociais",
                                "Eventos locais e características regionais"
                            ]
                        )
                    ], md=6, className="mb-4")
                ])
            ])
        ], style=estilo_card_principal),
        
        # Seção 3: Dicionário de Dados Interativo
        dbc.Card([
            dbc.CardBody([
                titulo_secao("Dicionário de Dados Interativo", "fa-book"),
                
                dbc.Row([
                    dbc.Col([
                        html.P([
                            "Selecione uma coluna para ver sua descrição detalhada. As colunas são provenientes de dois datasets: ",
                            html.Span("train_df", style={
                                'backgroundColor': AZUL_DESTAQUE,
                                'color': 'white',
                                'padding': '2px 8px',
                                'borderRadius': '4px',
                                'fontSize': '0.8rem',
                                'fontWeight': '600'
                            }),
                            " (dados de vendas) e ",
                            html.Span("store_df", style={
                                'backgroundColor': VERDE_DESTAQUE,
                                'color': 'white',
                                'padding': '2px 8px',
                                'borderRadius': '4px',
                                'fontSize': '0.8rem',
                                'fontWeight': '600'
                            }),
                            " (características das lojas)."
                        ], className="mb-3"),
                        
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Buscar coluna:", style={'fontWeight': '600', 'color': AZUL_ESCURO, 'marginBottom': '8px'}),
                                        dcc.Dropdown(
                                            id='dropdown-descricao-coluna',
                                            options=[{'label': col, 'value': col} for col in todas_colunas],
                                            placeholder="Selecione uma coluna...",
                                            clearable=True,
                                            style={
                                                'borderRadius': '8px',
                                                'border': f'2px solid {AZUL_DESTAQUE}',
                                                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
                                            }
                                        )
                                    ], md=6),
                                    
                                    dbc.Col([
                                        html.Div(
                                            id='saida-descricao-coluna',
                                            className="p-3 mt-3 mt-md-0",
                                            style={
                                                'backgroundColor': FUNDO_CINZA_CLARO,
                                                'borderRadius': '8px',
                                                'minHeight': '100px',
                                                'border': '1px solid #E0E0E0'
                                            }
                                        )
                                    ], md=6)
                                ])
                            ])
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 4px 10px rgba(0,0,0,0.08)',
                            'border': 'none',
                            'marginBottom': '25px'
                        })
                    ])
                ])
            ])
        ], style=estilo_card_principal),
        
        # Seção 4: Amostras dos Datasets
        dbc.Card([
            dbc.CardBody([
                titulo_secao("Amostras dos Datasets", "fa-table"),
                
                # Abas para alternar entre os datasets
                dbc.Tabs([
                    dbc.Tab([
                        dbc.Card([
                            dbc.CardBody([
            dbc.Row([
                dbc.Col([
                                        html.H5([
                                            html.I(className="fas fa-info-circle me-2", style={'color': AZUL_DESTAQUE}),
                                            "Sobre este Dataset"
                                        ], className="mb-3"),
                                        
                                        html.P([
                                            "Este dataset contém as vendas diárias das lojas Rossmann, incluindo informações sobre promoções, ",
                                            "feriados e status de abertura. É a principal fonte de dados para análise de vendas."
                                        ], className="mb-4"),
                                        
                    html.Div([
                        html.Div([
                                                html.Span("Linhas", className="fw-bold"),
                                                html.Span(f"{len(df_vendas_original):,}", className="ms-auto")
                                            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '8px'}),
                                            
                                            html.Div([
                                                html.Span("Colunas", className="fw-bold"),
                                                html.Span(f"{len(df_vendas_original.columns)}", className="ms-auto")
                                            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '8px'}),
                                            
                                            html.Div([
                                                html.Span("Período", className="fw-bold"),
                                                html.Span(f"{df_vendas_original['Date'].min().strftime('%Y-%m-%d')} - {df_vendas_original['Date'].max().strftime('%Y-%m-%d')}", className="ms-auto")
                                            ], style={'display': 'flex', 'justifyContent': 'space-between'})
                                        ], style={
                                            'backgroundColor': FUNDO_CINZA_CLARO,
                                            'padding': '15px',
                                            'borderRadius': '8px',
                                            'marginBottom': '20px'
                                        })
                                    ], md=4),
                                    
                                    dbc.Col([
                                        html.Div([
                                            html.H5("Amostra de Dados", className="mb-3"),
                                            
                                            html.Div(
                                                style={
                                                    'width': '100%',
                                                    'overflowX': 'auto',
                                                    'borderRadius': '8px',
                                                    'boxShadow': '0 2px 5px rgba(0,0,0,0.05)',
                                                    'backgroundColor': 'white',
                                                },
                                                children=[
                                                    html.Table(
                                                        # Header
                                                        [html.Thead(html.Tr([
                                                            html.Th(col, style={
                                                                'backgroundColor': '#f8f9fa',
                                                                'color': '#495057',
                                                                'fontWeight': '600',
                                                                'padding': '12px 15px',
                                                                'borderBottom': '2px solid #dee2e6',
                                                                'textAlign': 'left',
                                                                'whiteSpace': 'nowrap'
                                                            }) for col in df_vendas_original.columns
                                                        ]))] +
                                                        # Corpo da tabela
                                                        [html.Tbody([
                                                            html.Tr([
                                                                html.Td(
                                                                    # Formata data sem hora se a coluna for 'Date'
                                                                    df_vendas_original.iloc[i][col].strftime('%Y-%m-%d') if col == 'Date' and hasattr(df_vendas_original.iloc[i][col], 'strftime') else 
                                                                    df_vendas_original.iloc[i][col],
                                                                    style={
                                                                        'padding': '10px 15px',
                                                                        'borderBottom': '1px solid #dee2e6',
                                                                        'whiteSpace': 'nowrap'
                                                                    }
                                                                ) for col in df_vendas_original.columns
                                                            ], style={
                                                                'backgroundColor': 'white' if i % 2 == 0 else '#f8f9fa'
                                                            }) for i in range(min(5, len(df_vendas_original)))
                                                        ])],
                                                        style={
                                                            'borderCollapse': 'collapse',
                                                            'width': '100%',
                                                            'fontSize': '0.9rem',
                                                            'fontFamily': 'Arial, sans-serif'
                                                        },
                                                        className="table"
                                                    )
                                                ]
                                            )
                                        ])
                                    ], md=8)
                                ])
                            ])
                        ], style={
                            'border': 'none',
                            'borderRadius': '12px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.05)',
                            'marginTop': '15px'
                        })
                    ], label="Dataset de Vendas", tab_id="tab-vendas", 
                       tab_style={"marginRight": "0"}, 
                       label_style={"color": AZUL_DESTAQUE, "fontWeight": "600"}),
                    
                    dbc.Tab([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5([
                                            html.I(className="fas fa-info-circle me-2", style={'color': VERDE_DESTAQUE}),
                                            "Sobre este Dataset"
                                        ], className="mb-3"),
                                        
                                        html.P([
                                            "Este dataset contém informações detalhadas sobre cada loja Rossmann, incluindo tipo, sortimento, ",
                                            "distância de competidores e dados históricos de promoções."
                                        ], className="mb-4"),
                                        
                                        html.Div([
                                            html.Div([
                                                html.Span("Linhas", className="fw-bold"),
                                                html.Span(f"{len(df_caracteristicas_original):,}", className="ms-auto")
                                            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '8px'}),
                                            
                                            html.Div([
                                                html.Span("Colunas", className="fw-bold"),
                                                html.Span(f"{len(df_caracteristicas_original.columns)}", className="ms-auto")
                                            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '8px'}),
                                            
                                            html.Div([
                                                html.Span("Tipos de Loja", className="fw-bold"),
                                                html.Span(f"{df_caracteristicas_original['StoreType'].nunique()}", className="ms-auto")
                                            ], style={'display': 'flex', 'justifyContent': 'space-between'})
                                        ], style={
                                            'backgroundColor': FUNDO_CINZA_CLARO,
                                            'padding': '15px',
                                            'borderRadius': '8px',
                                            'marginBottom': '20px'
                                        })
                                    ], md=4),

                dbc.Col([
                    html.Div([
                                            html.H5("Amostra de Dados", className="mb-3"),
                                            
                                            html.Div(
                                                style={
                                                    'width': '100%',
                                                    'overflowX': 'auto',
                                                    'borderRadius': '8px',
                                                    'boxShadow': '0 2px 5px rgba(0,0,0,0.05)',
                                                    'backgroundColor': 'white',
                                                },
                                                children=[
                                                    html.Table(
                                                        # Header
                                                        [html.Thead(html.Tr([
                                                            html.Th(col, style={
                                                                'backgroundColor': '#f8f9fa',
                                                                'color': '#495057',
                                                                'fontWeight': '600',
                                                                'padding': '12px 15px',
                                                                'borderBottom': '2px solid #dee2e6',
                                                                'textAlign': 'left',
                                                                'whiteSpace': 'nowrap'
                                                            }) for col in df_caracteristicas_original.columns
                                                        ]))] +
                                                        # Corpo da tabela
                                                        [html.Tbody([
                                                            html.Tr([
                                                                html.Td(
                                                                    df_caracteristicas_original.iloc[i][col],
                                                                    style={
                                                                        'padding': '10px 15px',
                                                                        'borderBottom': '1px solid #dee2e6',
                                                                        'whiteSpace': 'nowrap'
                                                                    }
                                                                ) for col in df_caracteristicas_original.columns
                                                            ], style={
                                                                'backgroundColor': 'white' if i % 2 == 0 else '#f8f9fa'
                                                            }) for i in range(min(5, len(df_caracteristicas_original)))
                                                        ])],
                                                        style={
                                                            'borderCollapse': 'collapse',
                                                            'width': '100%',
                                                            'fontSize': '0.9rem',
                                                            'fontFamily': 'Arial, sans-serif'
                                                        },
                                                        className="table"
                                                    )
                                                ]
                                            )
                                        ])
                                    ], md=8)
                                ])
                            ])
                        ], style={
                            'border': 'none',
                            'borderRadius': '12px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.05)',
                            'marginTop': '15px'
                        })
                    ], label="Dataset de Características das Lojas", tab_id="tab-lojas", 
                       label_style={"color": VERDE_DESTAQUE, "fontWeight": "600"})
                ], id="tabs-datasets", active_tab="tab-vendas")
            ])
        ], style=estilo_card_principal),
        
        # Footer
        html.Footer([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.I(className="fas fa-info-circle me-2", style={'color': VERMELHO_ROSSMANN}),
                            "Este dashboard é parte de um projeto de análise de dados para estudo e demonstração."
                        ], style={'textAlign': 'center', 'color': CINZA_NEUTRO, 'margin': '0'})
                    ], width=12),
                    
                    dbc.Col([
                        html.Div([
                            html.A([
                                html.I(className="fab fa-github me-2"),
                                "Código-fonte"
                            ], href="#", style={'color': AZUL_ESCURO, 'textDecoration': 'none', 'marginRight': '15px'}),
                            
                            html.A([
                                html.I(className="fas fa-file-alt me-2"),
                                "Documentação"
                            ], href="#", style={'color': AZUL_ESCURO, 'textDecoration': 'none'})
                        ], style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '10px'})
                ], width=12)
            ])
            ], style={'backgroundColor': FUNDO_CINZA_CLARO, 'padding': '15px', 'borderRadius': '8px'})
        ], style={'marginTop': '30px', 'marginBottom': '20px'})

    ], fluid=True, className="py-4")

    return layout