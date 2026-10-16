# Diagnóstico de orientação de rótulo — versão independente.
#
# Não depende de nenhuma célula do notebook: define o que precisa.
# Cole numa célula nova de qualquer notebook, depois de montar o Drive,
# e rode. Só lê e mede; não altera nada.
#
# Roda em CPU. Com AMOSTRA = 150 leva menos de um minuto por modelo.

from pathlib import Path
import datetime
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path('/content/drive/MyDrive/TCC')
DADOS = BASE / 'dados'
SEMENTE = 42
AMOSTRA = 150          # basta para distinguir 99% de 1%
INCLUIR_CHECKPOINTS = True


def achar_csv(*nomes):
    for nome in nomes:
        for pasta in (DADOS, BASE):
            if (pasta / nome).exists():
                return pasta / nome
    raise FileNotFoundError('não encontrei nenhum de: ' + ', '.join(nomes))


# ---------------------------------------------------------------- modelos
pastas = []
for config in sorted(BASE.rglob('config.json')):
    pasta = config.parent
    if not ((pasta / 'model.safetensors').exists()
            or (pasta / 'pytorch_model.bin').exists()):
        continue
    if not INCLUIR_CHECKPOINTS and 'checkpoint' in pasta.name:
        continue
    marca = pasta / 'convencao_rotulo.json'
    convencao = json.loads(marca.read_text(encoding='utf-8')) if marca.exists() else None
    pastas.append((pasta,
                   datetime.datetime.fromtimestamp(config.stat().st_mtime),
                   convencao))

# O modelo final primeiro, depois do mais recente para o mais antigo: a
# resposta que interessa sai logo na primeira linha.
pastas.sort(key=lambda t: ('checkpoint' in t[0].name, -t[1].timestamp()))

print('=' * 72)
print('MODELOS ENCONTRADOS')
print('=' * 72)
for pasta, quando, convencao in pastas:
    print('{:<46} {}  {}'.format(
        str(pasta.relative_to(BASE)), quando.strftime('%d/%m %H:%M'),
        'convenção: ' + str(convencao.get('1')) if convencao else 'sem convenção'))
if not pastas:
    raise SystemExit('nenhum modelo encontrado em ' + str(BASE))

# ------------------------------------------------- conjunto de referência
df = pd.read_csv(achar_csv('corpus_preparado.csv', 'corpus_base.csv'))
df = df.dropna(subset=['texto'])
_, te_txt, _, te_y = train_test_split(
    df['texto'].values, df['rotulo'].values,
    test_size=0.2, random_state=SEMENTE, stratify=df['rotulo'].values)
te_txt, te_y = te_txt[:AMOSTRA], te_y[:AMOSTRA]

disp = 'cuda' if torch.cuda.is_available() else 'cpu'
print('\nconjunto de referência: {} notícias | dispositivo: {}'
      .format(len(te_txt), disp))
print('proporção de falsas na amostra: {:.1%}'.format(np.mean(te_y)))

# ------------------------------------------------------------- medição
print('\n' + '=' * 72)
print('ORIENTAÇÃO DE CADA MODELO')
print('=' * 72)
melhor = None
for pasta, quando, _ in pastas:
    nome = str(pasta.relative_to(BASE))
    try:
        tk = AutoTokenizer.from_pretrained(str(pasta))
        md = AutoModelForSequenceClassification.from_pretrained(str(pasta))
        md = md.eval().to(disp)
    except Exception as erro:
        print('{:<46} não carregou: {}'.format(nome, type(erro).__name__))
        continue

    with torch.no_grad():
        partes = []
        for i in range(0, len(te_txt), 32):
            ent = tk([str(t) for t in te_txt[i:i + 32]], truncation=True,
                     max_length=256, padding=True, return_tensors='pt').to(disp)
            partes.append(md(**ent).logits.argmax(-1).cpu().numpy())
    pred = np.concatenate(partes)

    direto = accuracy_score(te_y, pred)
    invertido = accuracy_score(te_y, 1 - pred)
    if direto >= invertido:
        veredito = 'CONVENÇÃO ATUAL  (1 = falsa)  — usar este'
    else:
        veredito = 'convenção antiga (1 = verdadeira)'
    print('\n{}  [{}]'.format(nome, quando.strftime('%d/%m %H:%M')))
    print('   direto {:>7.2%}   invertido {:>7.2%}   ->  {}'
          .format(direto, invertido, veredito))
    print('   matriz (linhas = real 0,1):',
          confusion_matrix(te_y, pred, labels=[0, 1]).tolist())

    if direto >= invertido and (melhor is None or direto > melhor[1]):
        melhor = (pasta, direto)

    del md, tk
    if disp == 'cuda':
        torch.cuda.empty_cache()

print('\n' + '=' * 72)
if melhor is None:
    print('NENHUM modelo está na convenção atual. O retreino precisa rodar de '
          'novo, com GPU, usando a Fase 5 nova.')
else:
    pasta, acerto = melhor
    print('USAR: {}   ({:.2%} na amostra)'.format(pasta.relative_to(BASE), acerto))
    print('\nPara fixar essa escolha, rode também a célula abaixo — ela grava')
    print('a declaração de convenção na pasta do modelo, e aí o notebook')
    print('passa a encontrá-lo sozinho:\n')
    print("    import json")
    print("    (Path(r'{}') / 'convencao_rotulo.json').write_text(".format(pasta))
    print("        json.dumps({'0': 'Verdadeira', '1': 'Falsa',")
    print("                    'texto': 'titulo + noticia'},")
    print("                   ensure_ascii=False, indent=1), encoding='utf-8')")
