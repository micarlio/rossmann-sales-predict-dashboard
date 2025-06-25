# dashboard/callbacks/__init__.py

# Importar as funções de registro de callbacks de cada módulo (imports relativos dentro do pacote)
from .callbacks_gerais import registrar_callbacks_gerais
from .callbacks_analise_preliminar import registrar_callbacks_analise_preliminar
from .callbacks_analise_3d import registrar_callbacks_analise_3d
from .callbacks_analise_lojas import registrar_callbacks_analise_lojas
from .callbacks_limpeza_dados import registrar_callbacks_limpeza_dados
from .callbacks_dashboard_geral import registrar_callbacks_dashboard_geral


def registrar_callbacks(aplicativo, dados):
    """Registra todos os callbacks da aplicação."""
    registrar_callbacks_gerais(aplicativo, dados)
    registrar_callbacks_analise_preliminar(aplicativo, dados)
    registrar_callbacks_dashboard_geral(aplicativo, dados)
    registrar_callbacks_analise_3d(aplicativo, dados)
    # Para a página de análise de lojas, usamos o dcc.Store em vez de passar o DataFrame diretamente
    registrar_callbacks_analise_lojas(aplicativo)
    # Callbacks para a página de Limpeza de Dados
    registrar_callbacks_limpeza_dados(aplicativo, dados)

    # Nota: A página 'Limpeza de Dados' é estática em sua maioria, não precisando de callbacks aqui.