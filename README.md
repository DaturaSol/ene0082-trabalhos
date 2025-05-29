#   Trabalho Computacional 2: Perceptron Multicamadas no Problema MNIST

## Instruções

**Atenção para Execução Local com CUDA:**

Caso deseje executar o projeto localmente utilizando uma GPU NVIDIA, este projeto foi configurado considerando o uso do CUDA.
É crucial que a versão do PyTorch instalada pelo Poetry seja compatível com a sua versão do driver NVIDIA e do CUDA Toolkit instalados em seu sistema.

*   O `pyproject.toml` deste projeto deve especificar a versão do PyTorch e sua respectiva compilação para CUDA.
*   Verifique a compatibilidade e as instruções de instalação do PyTorch com a sua versão do CUDA em: [PyTorch Get Started Locally](https://pytorch.org/get-started/locally/).
*   Certifique-se de ter o [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) apropriado instalado.

#### Instalação do Poetry

Recomenda-se utilizar uma versão específica do Poetry para garantir a reprodutibilidade do ambiente, conforme utilizado neste projeto. A instalação do Poetry pode ser feita via `pip`:

```bash
pip install poetry==2.1.1
```

```bash
curl -sSL https://install.python-poetry.org | python3 - --version 2.1.1
```

Embora a [documentação oficial do Poetry](https://python-poetry.org/docs/#installation) apresente outros métodos de instalação (como via `pipx` ou scripts de instalação específicos), o comando `pip` acima é direto para obter a versão `2.1.1` especificada. Certifique-se de que seu `pip` esteja associado à instalação Python correta.

#### Configurando o Ambiente e Instalando Dependências

Após instalar o Poetry e navegar até o diretório raiz do projeto no terminal, execute os seguintes comandos:

```bash
# Opcional: Configura o Poetry para criar o ambiente virtual dentro da pasta do projeto (em um diretório .venv)
poetry config virtualenvs.in-project true

# Cria o ambiente virtual (se não existir) e instala todas as dependências listadas no pyproject.toml
poetry install
```

Para ativar o ambiente virtual criado pelo Poetry:
```bash
poetry env activate
```

O comando `poetry install` já cuida da instalação das dependências. A flag `--no-root` (mencionada anteriormente) geralmente é usada quando se está desenvolvendo uma biblioteca e não se quer instalar o projeto atual como um pacote editável, o que pode não ser necessário para este tipo de projeto.

#### Sobre a Execução no Google Colab

Recomenda-se também testar a execução do projeto no Google Colab, que oferece ambientes com GPU. Contudo, podem surgir desafios ao tentar replicar exatamente a mesma versão do CUDA utilizada localmente ou configurada via Poetry, pois o Colab gerencia suas próprias versões de CUDA e drivers. Geralmente, o PyTorch instalado no Colab já vem com suporte a CUDA, mas pode ser uma versão diferente da especificada para desenvolvimento local.