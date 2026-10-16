# Mede o vazamento do veredito da checagem pelo título.
#
# Independente do notebook. Cole numa célula, depois de montar o Drive.
# Só lê e mede; não altera nada. Roda em CPU, em menos de um minuto.

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE = Path('/content/drive/MyDrive/TCC')
DADOS = BASE / 'dados'
SEMENTE = 42


def achar_csv(*nomes):
    for nome in nomes:
        for pasta in (DADOS, BASE):
            if (pasta / nome).exists():
                return pasta / nome
    raise FileNotFoundError('não encontrei: ' + ', '.join(nomes))


prep = pd.read_csv(achar_csv('corpus_preparado.csv', 'corpus_base.csv'))
bruto = pd.read_csv(achar_csv('fakerecogna_bruto.csv', 'FakeRecogna.csv'))
if 'indice_origem' in prep.columns:
    bruto = bruto.loc[prep['indice_origem'].to_numpy()].reset_index(drop=True)

col = {c.strip().lower(): c for c in bruto.columns}
TITULO = col.get('titulo') or col.get('título') or col.get('title')
NOTICIA = col.get('noticia') or col.get('notícia') or col.get('news')

titulo = bruto[TITULO].fillna('').astype(str)
noticia = bruto[NOTICIA].fillna('').astype(str)
classe = prep['classe'].to_numpy()
rotulo = prep['rotulo'].to_numpy()
e_falsa = classe == 'Falsa'

# ------------------------------------------------- 1. marcas de veredito
# Expressões que a agência usa para anunciar que a alegação é falsa.
MARCAS = {
    '#boato': r'#\s*boato',
    '#fake': r'#\s*fake',
    'é fake': r'\bé\s+#?fake\b',
    'é falso/falsa': r'\bé\s+fals[oa]\b',
    'é mentira': r'\bé\s+mentira\b',
    'boato (solto)': r'\bboato s?\b',
    'fake news': r'\bfake\s+news\b',
    'não é verdade': r'\bnão\s+é\s+verdade\b',
    'enganoso': r'\bengano[sa]',
    'verificamos/checamos': r'\b(verificamos|checamos|desmentid)',
}

print('=' * 76)
print('MARCAS DE VEREDITO NO TÍTULO')
print('=' * 76)
print('{:<22} {:>12} {:>14} {:>12}'.format(
    'marca', 'em falsas', 'em verdadeiras', 'diferença'))
qualquer = np.zeros(len(titulo), dtype=bool)
baixo = titulo.str.lower()
for nome, padrao in MARCAS.items():
    bate = baixo.str.contains(padrao, regex=True, na=False).to_numpy()
    qualquer |= bate
    pf, pv = bate[e_falsa].mean(), bate[~e_falsa].mean()
    print('{:<22} {:>11.1%} {:>14.1%} {:>11.1f} p.p.'
          .format(nome, pf, pv, 100 * (pf - pv)))

print('-' * 76)
print('{:<22} {:>11.1%} {:>14.1%} {:>11.1f} p.p.'
      .format('QUALQUER MARCA', qualquer[e_falsa].mean(),
              qualquer[~e_falsa].mean(),
              100 * (qualquer[e_falsa].mean() - qualquer[~e_falsa].mean())))

# Regra ingênua: sem aprendizado nenhum, só procurar a marca no título.
acerto_regra = accuracy_score(rotulo, qualquer.astype(int))
print('\nAcurácia de uma regra SEM APRENDIZADO ("título tem marca -> falsa"): '
      '{:.2%}'.format(acerto_regra))
print('É o piso que qualquer modelo com acesso ao título já recebe de graça.')

# ------------------------------------------- 2. ablação: o que carrega o sinal
print('\n' + '=' * 76)
print('ABLAÇÃO — de onde vem o desempenho')
print('=' * 76)

if 'particao' in prep.columns:
    treino = (prep['particao'] == 'treino').to_numpy()
else:
    from sklearn.model_selection import train_test_split
    pos = np.arange(len(prep))
    pos_tr, _ = train_test_split(pos, test_size=0.2, random_state=SEMENTE,
                                 stratify=rotulo)
    treino = np.zeros(len(prep), dtype=bool)
    treino[pos_tr] = True

def sem_marcas(texto_serie):
    limpo = texto_serie
    for padrao in MARCAS.values():
        limpo = limpo.str.replace(padrao, ' ', regex=True, case=False)
    return limpo

variantes = {
    'só o título': titulo,
    'só a notícia (corpo)': noticia,
    'título + notícia (atual)': titulo + ' ' + noticia,
    'título sem marcas + notícia': sem_marcas(titulo) + ' ' + noticia,
}

print('{:<32} {:>10} {:>12}'.format('entrada', 'acurácia', 'erros/2381'))
for nome, serie in variantes.items():
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
    X_tr = vec.fit_transform(serie[treino])
    X_te = vec.transform(serie[~treino])
    lr = LogisticRegression(max_iter=1000, random_state=SEMENTE)
    lr.fit(X_tr, rotulo[treino])
    pred = lr.predict(X_te)
    acerto = accuracy_score(rotulo[~treino], pred)
    print('{:<32} {:>9.2%} {:>12}'.format(
        nome, acerto, int((pred != rotulo[~treino]).sum())))

# ------------------------------------------------- 3. exemplos de título
print('\n' + '=' * 76)
print('TÍTULOS DE NOTÍCIAS FALSAS COM MARCA DE VEREDITO (amostra)')
print('=' * 76)
com_marca = titulo[e_falsa & qualquer]
for t in com_marca.sample(min(8, len(com_marca)), random_state=7):
    print('  •', ' '.join(str(t).split())[:150])

print('\n' + '=' * 76)
print("""COMO LER

A diferença de "QUALQUER MARCA" entre as classes é o tamanho do vazamento.
A acurácia da regra sem aprendizado mostra quanto desse sinal um modelo
recebe de graça ao ver o título.

Na ablação, compare as quatro linhas. Se "só o título" já chega perto do
desempenho de "título + notícia", o título carrega quase todo o sinal. Se
"título sem marcas + notícia" cai bastante em relação a "título + notícia",
o que o título carrega é o veredito da agência, não conteúdo da notícia.

Me mande esta saída inteira.""")
