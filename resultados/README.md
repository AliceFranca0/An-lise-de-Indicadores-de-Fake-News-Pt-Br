# Resultados

Esta pasta recebe o que a execução dos notebooks produz. Ela vai vazia no
repositório porque os arquivos são gerados, não escritos à mão.

Depois de rodar as oito fases, copie de `MyDrive/TCC/`:

- `dados/` — as tabelas de métricas (`metricas_*.csv`, `deteccao_sinteticas.csv`,
  `truncamento.csv`, `taxas_lematizacao.csv`, `vazamento_titulo.csv`,
  `termos_discriminantes.csv`, `agencias.csv`, `historico_bertimbau.csv`)
- `figuras/` — os 20 PNGs em 400 dpi

Não copie para cá o corpus, as predições linha a linha nem os modelos: são
grandes e reproduzíveis. O `.gitignore` já bloqueia `*.csv` fora de
`resultados/dados/`, para que um arquivo de dados não entre por engano.

A Fase 8 tem uma célula de resumo que imprime todos esses números de uma vez,
útil para conferir o que foi copiado.
