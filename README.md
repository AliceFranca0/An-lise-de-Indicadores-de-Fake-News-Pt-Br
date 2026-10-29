# Detecção de Fake News em Português: vazamentos de informação e descasamento de pré-processamento no corpus FakeRecogna

Trabalho de Conclusão de Curso — Tecnologia em Análise e Desenvolvimento de Sistemas
Instituto Federal do Paraná (IFPR), Campus Paranaguá

**Autora:** Alice Vitoria Cordeiro França · **Orientação:** Prof. Wagner Rodrigo Weinert

Auditoria do corpus FakeRecogna e avaliação comparativa de três detectores de
notícias falsas em português — Regressão Logística, Random Forest e BERTimbau —
diante de notícias falsas e verdadeiras geradas por modelo de linguagem, com a
forma e o comprimento do texto controlados.

## Resultados

### Desempenho no conjunto de teste do FakeRecogna (2.381 notícias)

| Modelo | Taxa de acerto | Precisão | Revocação | F1 | Erros |
|---|---|---|---|---|---|
| Regressão Logística | 93,99% | 94,25% | 93,70% | 93,97% | 143 |
| Random Forest | 92,27% | 92,06% | 92,52% | 92,29% | 184 |
| BERTimbau (modelo de referência) | 97,27% | 96,68% | 97,90% | 97,29% | 65 |

Em nove treinamentos do BERTimbau (quatro execuções e cinco sementes), a taxa
de acerto no teste ficou entre 97,27% e 97,69%.

### Detecção de 147 notícias falsas sintéticas, nas duas formas de texto

| Modelo | Texto natural | Texto lematizado |
|---|---|---|
| Regressão Logística | 65,31% | 79,59% |
| Random Forest | 45,58% | 48,98% |
| BERTimbau | 1,36% | 99,32% |

Nos nove treinamentos, o BERTimbau detectou entre 0% e 22,45% em texto natural
e entre 97,96% e 100% em texto lematizado.

### Taxa de acerto balanceada, com as 147 verdadeiras sintéticas incluídas

| Modelo | Texto natural | Texto lematizado |
|---|---|---|
| Regressão Logística | 80,95% | 89,12% |
| Random Forest | 66,33% | 69,73% |
| BERTimbau | 50,68% | 96,26% |

Valores do modelo de referência. Nas cinco sementes do estudo de variação, o
BERTimbau ficou entre 50,00% e 55,44% em texto natural e entre 87,41% e 92,52%
em texto lematizado — faixa que contém o valor da Regressão Logística.

## O que o trabalho encontrou

**Dois vazamentos de informação no corpus.** Marcas do veredito da checagem
(`#boato`, `é fake`, `é falso`) aparecem em 66,07% dos títulos falsos e em
0,50% dos verdadeiros; uma regra sem aprendizado nenhum acerta 82,78% do
corpus só com isso. Com título e corpo, a Regressão Logística chega a 95,88%;
sem as marcas, cai para 93,70%, abaixo dos 93,99% obtidos só com o corpo —
todo o ganho do título era o vazamento. O subtítulo está preenchido em 75,3%
dos registros falsos e em 18,5% dos verdadeiros. Os modelos recebem apenas o
corpo da notícia.

**O corpo das notícias é distribuído lematizado, e a documentação não diz
isso.** 94,54% dos *tokens* da classe Falsa e 91,11% dos da Verdadeira já estão
na forma de lema, contra cerca de 60% no português corrente. O BERTimbau
treinado nesse material reconhece a desinformação na forma do corpus e deixa de
reconhecê-la quando o texto chega como um usuário o escreveria. A falha é
silenciosa: nenhuma métrica interna a acusa. Por isso a aplicação da Fase 8
lematiza a entrada antes de consultar os modelos.

**Com a forma controlada, a ordem dos modelos se mantém.** Na detecção das
falsas sintéticas, a ordem do corpus (BERTimbau, Regressão Logística, Random
Forest) se preserva, e a hipótese inicial — de que o modelo de maior
capacidade seria o mais vulnerável — não se sustentou. Em texto natural, a
ordem pela taxa de acerto balanceada muda e o BERTimbau cai para o último
lugar: uma avaliação sem controle de forma chegaria à conclusão oposta.

**Cada modelo falha por uma razão distinta.** No BERTimbau, o efeito da forma
do texto é de 97,96 pontos percentuais. No Random Forest, cortar os textos
sintéticos em 61 palavras (a mediana da classe Falsa) eleva a detecção de
48,98% para 68,71%. O comprimento está associado às decisões dos classificadores
clássicos; o corte também remove conteúdo, então o efeito é tratado como
associação, não como causa.

**A variação entre treinamentos está fora do corpus.** Entre cinco sementes, a
taxa de acerto no teste variou 0,25 ponto percentual, enquanto a especificidade
sobre as verdadeiras sintéticas em texto lematizado variou 11,56 pontos. Uma
conclusão baseada em um único treinamento pode atribuir ao modelo o que é
efeito do acaso.

## Estrutura

```
notebooks/            as nove fases, na ordem de execução
src/estilo_sbc.py     padrão visual único das figuras (template SBC)
resultados/dados/     métricas agregadas gravadas pelas fases (CSV)
resultados/figuras/   figuras geradas pelas fases (PNG, 400 dpi)
```

| Fase | O que faz | GPU |
|---|---|---|
| 1 | Caracterização do corpus, convenção de rótulo, partição fixa, vazamento do título, duplicatas | — |
| 2 | Recorte temporal e vocabulário discriminante | — |
| 3 | TF-IDF, Regressão Logística e Random Forest; ablação do título; versões do ambiente | — |
| 4 | Procedência das checagens por agência | — |
| 5 | Ajuste fino e reavaliação do BERTimbau | sim |
| 6 | Conjunto sintético: caracterização e versão lematizada | — |
| 5B | Estudo de sementes do BERTimbau (rodar depois da Fase 6) | sim |
| 7 | Robustez: delineamento fatorial, curva de truncamento, verdadeiras sintéticas, LIME | sim |
| 8 | Fluxograma do método, aplicação web e resumo numérico | — |

As fases são encadeadas: cada uma lê os arquivos gravados pelas anteriores em
`MyDrive/TCC/`.

## Execução

Os notebooks foram escritos para o Google Colab, com os dados no Google Drive
em `MyDrive/TCC/dados/`, os modelos em `MyDrive/TCC/modelos/` e as figuras em
`MyDrive/TCC/figuras/`.

Três chaves controlam o que é refeito:

- `RETREINAR = False` (Fase 5) — reavalia o modelo salvo em vez de treinar de
  novo. O ajuste fino em GPU não é determinístico: um novo treinamento gera um
  modelo diferente, dentro da faixa descrita acima.
- `REGERAR = False` (Fase 6) — usa o conjunto sintético já gerado.
- A Fase 5B retoma de onde parou: sementes já avaliadas não são treinadas de novo.

Fora do Colab:

```
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

### Ambiente dos resultados

Google Colab com GPU NVIDIA Tesla T4, Python 3.13, pandas 2.2.3, NumPy 2.1.3,
scikit-learn 1.6.1, NLTK 3.9.1, Matplotlib 3.10.0, PyTorch 2.11.0,
Transformers 5.20.2, spaCy 3.8.16 (modelo `pt_core_news_sm`), Starlette 1.9.0
e Uvicorn 0.54.0. Modelo de base: `neuralmind/bert-base-portuguese-cased`.

### Dados não incluídos

- **Corpus FakeRecogna.** É distribuído pelos autores originais
  ([Hugging Face](https://huggingface.co/datasets/recogna-nlp/FakeRecogna)).
  Coloque o arquivo em `MyDrive/TCC/dados/` com o nome `fakerecogna_bruto.csv`
  ou `FakeRecogna.csv` antes de rodar a Fase 1.
- **Modelo BERTimbau ajustado.** Não está aqui, por tamanho. A Fase 5 com
  `RETREINAR = True` treina um novo.
- **Conjunto de notícias sintéticas.** É material de pesquisa: permanece
  restrito ao ambiente do estudo, cada registro é identificado como sintético e
  nada dele é publicado, nem como exemplo nas saídas dos notebooks.

### Chave de API

A Fase 6 usa a API do Gemini apenas para regerar o conjunto sintético, e a
geração vem desligada por padrão. A chave é lida do painel de segredos do
Colab, nunca do código:

```python
from google.colab import userdata
userdata.get('GEMINI_API_KEY')
```

Nenhuma chave deve ser escrita nos notebooks: este repositório é público.

## Reprodutibilidade

- A partição de teste (20%, estratificada, semente 42) é gravada como coluna no
  próprio arquivo de dados, e não recalculada a cada fase. A validação (10% do
  restante) seleciona a época do BERTimbau; o teste é avaliado uma única vez.
- Uma assinatura do corpus e a convenção de rótulo (`1 = Falsa`) ficam gravadas
  junto ao modelo e são conferidas sempre que ele é carregado.
- Duas execuções do BERTimbau com o mesmo código, a mesma semente e as mesmas
  partições diferiram em 0,38 ponto percentual no teste e em 21 pontos na
  detecção em texto natural. Por isso o trabalho reporta faixas entre
  treinamentos e não afirma nada sobre a direção preferencial do erro do
  modelo neural.

## Aplicação web

A Fase 8 sobe, no próprio Colab, uma API em Starlette executada por Uvicorn e
uma interface em HTML com Bootstrap 5 e Chart.js. A aplicação lematiza o texto
com o spaCy, consulta os três modelos e mostra também o que cada um responderia
sem a lematização. As probabilidades não são calibradas: a ferramenta
demonstra o método e não deve ser usada como verificador em produção.

## Como citar

França, A. V. C. e Weinert, W. R. (2026) "Detecção de Fake News em Português:
vazamentos de informação e descasamento de pré-processamento no corpus
FakeRecogna". Trabalho de Conclusão de Curso (Tecnologia em Análise e
Desenvolvimento de Sistemas), Instituto Federal do Paraná, Paranaguá.

## Referências principais

Garcia, G. L., Afonso, L. C. S. e Papa, J. P. (2022). FakeRecogna: A New
Brazilian Corpus for Fake News Detection. *PROPOR*, LNCS 13208, p. 57–67.

Souza, F., Nogueira, R. e Lotufo, R. (2020). BERTimbau: Pretrained BERT Models
for Brazilian Portuguese. *BRACIS*, p. 403–417.

Geirhos, R. et al. (2020). Shortcut Learning in Deep Neural Networks.
*Nature Machine Intelligence*, v. 2, p. 665–673.

Moraes Silva, R. et al. (2025). Fake News Detection in Portuguese Under Large
Language Model-Generated Content. *Journal of the Brazilian Computer Society*,
v. 31, n. 1, p. 1150–1167.

Haroon, M. A. et al. (2026). When Benchmarks Mislead: Shortcut Learning, Length
Confounds, and the Limits of Cross-Dataset Generalization in Multilingual Fake
News and Sarcasm Detection. arXiv:2607.14131.