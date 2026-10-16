# Detecção de fake news em português: vazamentos de informação e descasamento de pré-processamento no FakeRecogna

Trabalho de Conclusão de Curso — Tecnologia em Análise e Desenvolvimento de Sistemas
Instituto Federal do Paraná, Campus Paranaguá

Auditoria do corpus FakeRecogna e avaliação comparativa de três detectores de
notícias falsas em português — Regressão Logística, Random Forest e BERTimbau —
frente a desinformação gerada por modelo de linguagem.

## Resultados

Desempenho sobre o conjunto de teste do corpus (2.381 notícias):

| Modelo | Acurácia | Precisão | Revocação | F1 | Erros |
|---|---|---|---|---|---|
| Regressão Logística | 93,99% | 94,25% | 93,70% | 93,97% | 143 |
| Random Forest | 92,27% | 92,06% | 92,52% | 92,29% | 184 |
| BERTimbau | 97,52% | 96,85% | 98,24% | 97,54% | 59 |

Detecção sobre 147 notícias falsas geradas por modelo de linguagem, nas duas
formas de texto:

| Modelo | Texto natural | Texto lematizado | Custo do descasamento |
|---|---|---|---|
| Regressão Logística | 65,31% | 79,59% | −14,29 p.p. |
| Random Forest | 45,58% | 48,98% | −3,40 p.p. |
| BERTimbau | 0,00% | 100,00% | −100,00 p.p. |

### O que o trabalho encontrou

**O corpo das notícias do FakeRecogna é distribuído em forma lematizada.**
93% dos tokens já coincidem com o próprio lema, contra 60% do português
corrente. Um modelo treinado sobre esse material opera corretamente sobre
texto lematizado e falha por completo sobre texto natural: o BERTimbau detecta
todas as 147 notícias sintéticas na forma do treino e nenhuma delas em
português corrente, atribuindo probabilidade média de 0,02 à classe falsa.
A falha é silenciosa — nenhuma métrica interna acusa o problema.

Consequência prática: qualquer aplicação construída sobre este corpus que
entregue ao modelo o texto digitado por um usuário, sem replicar a
lematização, classificará como verdadeira toda notícia falsa que receber. A
aplicação da Fase 8 lematiza a entrada por essa razão.

**Dois vazamentos de informação no corpus.** Marcas do veredito da checagem
(`#boato`, `é fake`, `é falso`) aparecem em 66,07% dos títulos falsos e em
0,50% dos verdadeiros — uma regra sem aprendizado algum atinge 82,73% de
acurácia só com isso. O campo de subtítulo está preenchido em 75,3% dos
registros verdadeiros e em 18,5% dos falsos. Ambos foram removidos antes do
treino; os modelos recebem apenas o corpo da notícia.

**Não há inversão de hierarquia.** Com a forma do texto sob controle, a ordem
de mérito no corpus (BERTimbau, Regressão Logística, Random Forest) se
preserva sobre o material sintético. A hipótese inicial do trabalho — de que
o modelo de maior capacidade seria o menos robusto — foi refutada pelos
próprios dados.

**A fragilidade do Random Forest é confundidor de comprimento.** Truncados os
textos sintéticos na mediana da classe falsa (61 palavras), sua detecção sobe
de 48,98% para 68,71%.

## Estrutura

```
notebooks/   as oito fases, na ordem de execução
src/         estilo_sbc.py — padrão visual único das figuras
scripts/     diagnósticos avulsos, independentes dos notebooks
resultados/  CSVs e figuras produzidos pela execução (ver abaixo)
```

As fases são encadeadas: cada uma lê os arquivos gravados pelas anteriores.

| Fase | O que faz |
|---|---|
| 1 | Caracterização do corpus, dedução da convenção de rótulo, partição fixa, medição do vazamento do título |
| 2 | Recuperação temporal e análise de vocabulário |
| 3 | TF-IDF e classificadores clássicos |
| 4 | Procedência por agência de checagem |
| 5 | Ajuste fino do BERTimbau |
| 6 | Geração das notícias sintéticas e preparo das duas formas de texto |
| 7 | Avaliação de robustez, truncamento e explicabilidade |
| 8 | Fluxograma, aplicação web e resumo numérico |

## Execução

Os notebooks foram escritos para o Google Colab, com os dados no Drive em
`MyDrive/TCC/`. As Fases 5 e 7 exigem GPU.

```
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

### Dados não incluídos

O corpus FakeRecogna não acompanha este repositório. Ele é distribuído pelos
autores originais; veja a referência abaixo. Coloque o arquivo em
`MyDrive/TCC/dados/` antes de rodar a Fase 1.

O modelo BERTimbau ajustado também não está aqui, por tamanho. A Fase 5 o
reproduz em cerca de 40 minutos em GPU T4.

O conjunto de notícias sintéticas é material de laboratório: permanece
restrito ao ambiente de pesquisa, cada registro é identificado como sintético
na origem e nada disso é publicado como conteúdo jornalístico.

### Chave de API

A Fase 6 usa a API do Gemini para gerar as notícias sintéticas, e a geração
vem desligada por padrão (`REGERAR = False`). A chave é lida do painel de
segredos do Colab, nunca do código:

```python
userdata.get('GEMINI_API_KEY')
```

Nenhuma chave deve ser escrita nos notebooks — este repositório é público e
uma chave no código fica exposta no primeiro commit.

## Reprodutibilidade

A partição treino/teste é gravada como coluna no próprio arquivo de dados, e
não recalculada a cada execução: qualquer alteração na rotulagem mudaria a
estratificação e faria um modelo antigo ter visto parte do novo conjunto de
teste. Uma assinatura criptográfica do corpus é verificada no carregamento do
modelo, e a convenção de rótulo fica gravada junto aos pesos.

Duas fontes de variação permanecem. O Random Forest usa paralelismo não
determinístico e varia cerca de 0,25 p.p. entre execuções. O ajuste fino do
BERTimbau varia cerca de 0,13 p.p. em acurácia, com redistribuição dos erros
entre as classes — por isso o trabalho não afirma nada sobre a direção
preferencial do erro do modelo neural.

## Referências principais

Garcia, G. L., Afonso, L. C. S. e Papa, J. P. (2022). FakeRecogna: A New
Brazilian Corpus for Fake News Detection. *PROPOR*, LNCS 13208, p. 57–67.

Souza, F., Nogueira, R. e Lotufo, R. (2020). BERTimbau: Pretrained BERT Models
for Brazilian Portuguese. *BRACIS*, p. 403–417.

Geirhos, R. et al. (2020). Shortcut Learning in Deep Neural Networks.
*Nature Machine Intelligence*, v. 2, p. 665–673.

Haroon, M. A. et al. (2026). When Benchmarks Mislead: Shortcut Learning,
Length Confounds, and the Limits of Cross-Dataset Generalization.
arXiv:2607.14131.
