# BERT4Text

**Finetuning de modelos BERT para classificação de laudos médicos**

![UFABC Logo](assets/logotipo-ufabc-extenso.png)

Universidade Federal do ABC - Bacharelado em Ciência e Tecnologia<br />
Inteligência Artificial 2024/Q3

Lenin Cristi<br />
lenin.cristi@aluno.ufabc.edu.br

## Resumo

**Resumo: Este artigo apresenta uma visão geral teórica dos modelos BERT e suas variantes, destacando seus fundamentos e aplicações. Além disso, fornece um guia prático que pode ser usado como base para realizar o fine-tuning desses modelos em tarefas de classificação de texto demonstrando sua aplicabilidade em contextos clínicos.**

**Abstract: This article presents a theoretical overview of BERT models and their variants, highlighting their foundations and applications. Additionally, it provides a practical guide that can serve as a basis for fine-tuning these models on text classification tasks, demonstrating their applicability in clinical contexts.**

## Finetuning do modelo BERT

O passo a passo do finetuning está no notebook [BERTFinetuning](BERTFinetuning.ipynb).

## Dados utilizados

O conjunto de dados utilizado como base foi o [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) ele é amplamente utilizado para análise e classificação de câncer de mama. Ele contém medições computacionais de características extraídas de imagens de massas celulares em exames de mamografia. As características foram calculadas a partir de imagens digitalizadas de aspirações por agulha fina (FNA), sendo usadas para prever se uma massa é maligna (câncer) ou benigna (não cancerígena). Estes dados foram disponibilizados no repositório [UCI Machine Learning](https://archive.ics.uci.edu/) e originalmente coletados pelo Dr. William H. Wolberg da University of Wisconsin Hospitals, Madison.

## Preparação dos dados

A preparação dos dados está no notebook [PrepareData](PrepareData.ipynb).

Os dados originais foram aumentados sinteticamente somente para fins da demonstração do passo a passo do finetuning.

Para abordagens de **mundo real**, são necessários métodos de aumento mais robustos utilizando por exemplo

- Modelos LLM para geração de dados sintéticos (como no exemplo curto [SinteticData](SinteticData.ipynb) que usa Llama 3.2) baseados em dados de mundo real (como o Wisconsin Dataset usado aqui) e **revisados** por corpo médico regularmente registrado.

- Dados anotados **diretamente** por corpo médico regularmente registrado.

É preciso ter em mente também que modelos de aprendizado de máquina implementados na área de saúde **sempre** devem:

- Ter sua acurácia acompanhada por retorno humano (Human Feedback)

- Ter explicabilidade (Explainability)

- Ter critérios de honestidade definidos (Fairness)

- (Para dados de Brasileiros) Obedecer a LGPD para treinamento do modelo e inferência dos dados

- (Na União Européia) Obedecer a GPDR para treinamento do modelo e inferência dos dados

- (Nos Estados Unidos) Obedecer a HIPAA para treinamento do modelo e inferência dos dados

## Anexo: Como gerar o ambiente para reproduzir os experimentos

### Utilizando Conda com o arquivo environments.yml neste repositório

Para criar o ambiente com os pacotes necessários a partir de arquivo

```bash
conda env create -f environment.yml
```

### Utilizando Conda

Para criar, ativar e instalar os pacotes necessários num ambiente

```bash
conda create -n health python=3.11
```

```bash
conda activate health
```

```bash
conda install numpy matplotlib scikit-learn ipykernel pandas -c defaults -c conda-forge
```

### Utilizando Pip

Crie e ative um ambiente usando pip ou pyenv antes e instale os seguintes pacotes

```bash
pip install numpy matplotlib scikit-learn ipykernel pandas
```

## Referências

### Papers

J. Devlin, M. Chang, K. Lee, and K. Toutanova<br />
**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2019)**, North American Chapter of the Association for Computational Linguistics<br />
https://arxiv.org/abs/1810.04805

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin<br />
**Attention is All You Need (2017)**, Advances in Neural Information Processing Systems 30 (NIPS 2017)<br />
https://arxiv.org/abs/1706.03762

M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer<br />
**Deep contextualized word representations (2018)**, Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)<br />
https://arxiv.org/abs/1802.05365

A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever<br />
**Improving Language Understanding by Generative Pre-Training (2018)**<br />
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

### Dados

**Breast Cancer Wisconsin (Diagnostic)**<br />
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### Técnicas

**A Complete Guide to BERT with Code**<br />
https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11

Fine-tuning BERT for Text Classification: A Step-by-Step Guide<br />
https://medium.com/@coderhack.com/fine-tuning-bert-for-text-classification-a-step-by-step-guide-1a1c5f8e8ae1

Mastering BERT: A Comprehensive Guide from Beginner to Advanced in Natural Language Processing (NLP)<br />
https://medium.com/@shaikhrayyan123/a-comprehensive-guide-to-understanding-bert-from-beginners-to-advanced-2379699e2b51

### Modelos

Hugging Face<br />
**Fine-Tuned BERT Models (2024)**, HuggingFace.co<br />
https://huggingface.co/models?sort=trending&search=BERT

Meta Llama 3.2<br />
https://ai.meta.com/blog/meta-llama-3/
https://github.com/meta-llama/llama3
https://huggingface.co/meta-llama
https://huggingface.co/meta-llama/Llama-3.2-3B
https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2
https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md

___

CMCC - Universidade Federal do ABC (UFABC) - Santo André - SP - Brasil
