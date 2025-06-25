# flake8: noqa

# ==============================================================================
# PONTO DE ENTRADA PARA OS LAYOUTS
# ==============================================================================
"""
Este módulo serve como ponto de entrada centralizado para todos os layouts e 
componentes compartilhados do dashboard. Facilita a importação organizada em 
outros módulos como app.py e callbacks.py.

Importa e expõe:
1. Componentes compartilhados (barra lateral, cards, etc.)
2. Layouts específicos de cada página
"""

# Importação de Componentes Compartilhados
from .componentes_compartilhados import (
    barra_lateral,
    armazenamento_estado_barra_lateral,
    criar_card_grafico,
    criar_card_grafico_3d,
    criar_botoes_cabecalho,
    criar_card_filtros,
    criar_card_filtros_3d,
    criar_card_filtros_analise_lojas
)

# Importação dos Layouts das Páginas
from .layout_contextualizacao import criar_layout_contextualizacao
from .layout_limpeza_dados import criar_layout_limpeza_dados
from .layout_analise_preliminar import criar_layout_analise_preliminar
from .layout_dashboard_geral import criar_layout_dashboard_analise
from .layout_analise_lojas import criar_layout_analise_lojas
from .layout_analise_3d import criar_layout_analise_3d
from .layout_previsao_vendas import criar_layout_previsao_vendas