# Instalação e Configuração do Ambiente Virtual

sudo apt update && sudo apt install python3-venv

# Navegue até a pasta desejada e execute:

python3 -m venv .venv
source .venv/bin/activate

# Para encerrar o ambiente:
deactivate

# Bibliotecas Necessárias - Gênio Digital
pip install numpy scipy pandas matplotlib shapely psutil pillow

	# 1. Rede Hidráulica

	Numpy - Computação científica no Python.
	Scipy (utilizada pelo numpy) - Complemento na operação eficiente com matrizes.
    Pandas - Organização e análise de dados no formato de tabela.
    Matplotlib - Geração de gráficos e visualização 2D.
    Shapely - Lógica da construção da rede hidráulica.

    * pip install numpy scipy pandas matplotlib shapely

	# Aliases de Importação Recomendados:
	import generate_graph_arrays as GeraGrafo
	import PlotaRede as PlotaRede

	# 2. Placa Térmica

	Numpy - Computação científica no Python.
	Scipy (utilizada pelo numpy) - Complemento na operação eficiente com matrizes.
    Pandas - Organização e análise de dados no formato de tabela.
    Matplotlib - Geração de gráficos e visualização 2D.
    Shapely - Lógica da construção da placa térmica.
	Psutil - Informações sobre processos e uso do sistema.
	Pillow - Biblioteca padrão para o processamento de imagens. Utiilzado como motor pelo matplotlib para gerar a animação do tópico 7.

    * pip install numpy scipy pandas matplotlib shapely psutil pillow
