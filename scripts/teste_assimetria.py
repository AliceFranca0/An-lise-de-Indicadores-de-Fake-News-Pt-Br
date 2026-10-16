# O FakeRecogna lematiza as duas classes do mesmo jeito?
#
# Indício que levantou a suspeita: os termos discriminantes da classe FALSA
# são artefatos de lematização ("umar", "segundar", "vezar", "grupar"),
# enquanto os da classe VERDADEIRA são formas flexionadas ("casos", "mortes",
# "afirmou"). Se o pré-processamento fosse igual nas duas, os dois lados
# teriam o mesmo tipo de forma.
#
# Se o pré-processamento for ASSIMÉTRICO, ele é por si só um rótulo: o modelo
# pode separar as classes pela assinatura do processamento, sem ler conteúdo
# nenhum. Seria o terceiro e maior vazamento do corpus.
#
# PARTE A  - mede a taxa de lematização de cada classe
# PARTE B  - reavalia os TRÊS modelos com as sintéticas lematizadas
# PARTE C  - refaz o truncamento sobre o texto lematizado
#
# Roda em CPU. Não altera nenhum arquivo do projeto.

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path('/content/drive/MyDrive/TCC')
DADOS = BASE / 'dados'
SEMENTE = 42

prep = pd.read_csv(DADOS / 'corpus_preparado.csv')
sint = pd.read_csv(DADOS / 'fake_news_sinteticas.csv')

col_corpus = next(c for c in ('texto', 'texto_limpo', 'noticia', 'corpo')
                  if c in prep.columns)
col_sint = next(c for c in ('texto_sintetico', 'texto', 'noticia')
                if c in sint.columns)
textos_sint = sint[col_sint].fillna('').astype(str).tolist()

print('corpus     :', len(prep), 'registros | coluna:', repr(col_corpus))
print('sintéticas :', len(textos_sint), '| coluna:', repr(col_sint))

import spacy
try:
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'spacy', 'download',
                    'pt_core_news_sm'], check=True)
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])


# ============================================ PARTE A: assimetria de forma
print('\n' + '=' * 72)
print('PARTE A - AS DUAS CLASSES TÊM O MESMO PRÉ-PROCESSAMENTO?')
print('=' * 72)
print('Mede, em cada classe, quantos tokens JÁ ESTÃO na forma de lema.')
print('Texto lematizado tem taxa alta; texto natural tem taxa baixa.')


def taxa_de_lema(textos, limite=400):
    """Proporção de tokens alfabéticos que já coincidem com seu lema."""
    iguais = total = 0
    for doc in nlp.pipe(textos[:limite], batch_size=32):
        for t in doc:
            if t.is_alpha and len(t.text) > 2:
                total += 1
                if t.text.lower() == t.lemma_.lower():
                    iguais += 1
    return iguais / total if total else float('nan')


amostra_f = prep[prep['rotulo'] == 1][col_corpus].fillna('').astype(str).tolist()
amostra_v = prep[prep['rotulo'] == 0][col_corpus].fillna('').astype(str).tolist()

taxa_f = taxa_de_lema(amostra_f)
taxa_v = taxa_de_lema(amostra_v)
taxa_s = taxa_de_lema(textos_sint)

print('\n  {:<42} {:>10}'.format('conjunto', 'taxa'))
print('  ' + '-' * 54)
print('  {:<42} {:>9.1%}'.format('corpus — classe FALSA', taxa_f))
print('  {:<42} {:>9.1%}'.format('corpus — classe VERDADEIRA', taxa_v))
print('  {:<42} {:>9.1%}'.format('sintéticas (português natural)', taxa_s))
print('  ' + '-' * 54)
print('  {:<42} {:>8.1f} p.p.'.format('diferença entre as classes',
                                      100 * (taxa_f - taxa_v)))

print('\n  amostra de texto de cada classe:')
print('    FALSA     :', ' '.join(amostra_f[0].split())[:150])
print('    VERDADEIRA:', ' '.join(amostra_v[0].split())[:150])

print()
if abs(taxa_f - taxa_v) > 0.08:
    print('  *** ASSIMETRIA CONFIRMADA ***')
    print('  As classes NÃO recebem o mesmo pré-processamento. A assinatura do')
    print('  processamento é, por si só, um sinal que separa as classes — um')
    print('  terceiro vazamento, e o mais grave, porque não há como removê-lo')
    print('  sem reprocessar o corpus inteiro a partir do texto original.')
else:
    print('  Sem assimetria relevante. As duas classes recebem o mesmo')
    print('  tratamento, e o problema do BERTimbau é só a diferença entre o')
    print('  corpus (lematizado) e as sintéticas (naturais).')


# ================================ PARTE B: os três modelos, forma casada
print('\n' + '=' * 72)
print('PARTE B - OS TRÊS MODELOS COM AS SINTÉTICAS LEMATIZADAS')
print('=' * 72)
print('Os clássicos sofrem do mesmo descasamento que o BERTimbau. A medida')
print('de robustez só é válida se as formas estiverem casadas.')

lematizadas = [' '.join(t.lemma_ for t in doc)
               for doc in nlp.pipe(textos_sint, batch_size=32)]

# --- clássicos: retreinados aqui, com a configuração dos notebooks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

treino = prep['particao'] == 'treino'
texto_corpus = prep[col_corpus].fillna('').astype(str)

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    vazias = stopwords.words('portuguese')
except Exception:
    vazias = None

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5,
                      stop_words=vazias)
X_tr = vec.fit_transform(texto_corpus[treino])
X_te = vec.transform(texto_corpus[~treino])
y_tr = prep.loc[treino, 'rotulo'].to_numpy()
y_te = prep.loc[~treino, 'rotulo'].to_numpy()

lr = LogisticRegression(max_iter=1000, random_state=SEMENTE).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=200, random_state=SEMENTE,
                            n_jobs=-1).fit(X_tr, y_tr)
print('\n  conferência (deve bater com a Fase 8):')
print('    LR no teste do corpus : {:.2%}  (esperado 93,99%)'
      .format(accuracy_score(y_te, lr.predict(X_te))))
print('    RF no teste do corpus : {:.2%}  (esperado 92,27%)'
      .format(accuracy_score(y_te, rf.predict(X_te))))

# --- BERTimbau
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

caminho = next(c.parent for c in sorted(BASE.rglob('config.json'))
               if (c.parent / 'model.safetensors').exists()
               or (c.parent / 'pytorch_model.bin').exists())
tok = AutoTokenizer.from_pretrained(str(caminho))
mod = AutoModelForSequenceClassification.from_pretrained(str(caminho)).eval()
disp = 'cuda' if torch.cuda.is_available() else 'cpu'
mod.to(disp)


def bert_detecta(lista, lote=16):
    saida = []
    for i in range(0, len(lista), lote):
        ent = tok(lista[i:i + lote], truncation=True, padding=True,
                  max_length=256, return_tensors='pt').to(disp)
        with torch.no_grad():
            saida.append(mod(**ent).logits.argmax(1).cpu().numpy())
    return (np.concatenate(saida) == 1).mean()


def taxas(lista):
    Xs = vec.transform(lista)
    return ((lr.predict(Xs) == 1).mean(),
            (rf.predict(Xs) == 1).mean(),
            bert_detecta(lista))


nat = taxas(textos_sint)
lem = taxas(lematizadas)

print('\n  {:<22} {:>12} {:>14} {:>12}'
      .format('modelo', 'natural', 'lematizado', 'variação'))
print('  ' + '-' * 62)
for nome, a, b in zip(('Regressão Logística', 'Random Forest', 'BERTimbau'),
                      nat, lem):
    print('  {:<22} {:>11.2%} {:>13.2%} {:>+11.2f} p.p.'
          .format(nome, a, b, 100 * (b - a)))

print('\n  acurácia de referência no corpus, para comparar:')
print('    LR 93,99%  |  RF 92,27%  |  BERTimbau 97,65%')


# ================================ PARTE C: truncamento, agora com forma casada
print('\n' + '=' * 72)
print('PARTE C - TRUNCAMENTO SOBRE O TEXTO LEMATIZADO')
print('=' * 72)

alvo = int(prep[prep['rotulo'] == 1][col_corpus]
           .fillna('').astype(str).str.split().str.len().median())
cortadas = [' '.join(t.split()[:alvo]) for t in lematizadas]
print('cortando na mediana da classe Falsa: {} palavras'.format(alvo))

tru = taxas(cortadas)
print('\n  {:<22} {:>12} {:>14} {:>12}'
      .format('modelo', 'completo', 'truncado', 'variação'))
print('  ' + '-' * 62)
for nome, a, b in zip(('Regressão Logística', 'Random Forest', 'BERTimbau'),
                      lem, tru):
    print('  {:<22} {:>11.2%} {:>13.2%} {:>+11.2f} p.p.'
          .format(nome, a, b, 100 * (b - a)))

# ------------------------------------------------------------ guardar
saida = pd.DataFrame({
    'modelo': ['Regressão Logística', 'Random Forest', 'BERTimbau'],
    'corpus': [0.9399, 0.9227, 0.9765],
    'sintetica_natural': nat,
    'sintetica_lematizada': lem,
    'sintetica_lematizada_truncada': tru,
})
saida.to_csv(DADOS / 'robustez_forma_casada.csv', index=False)
print('\ngravado: dados/robustez_forma_casada.csv')

print('\n' + '=' * 72)
print('Me mande esta saída inteira.')
