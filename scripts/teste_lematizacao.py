# BERTimbau detecta 0 das 147 sintéticas. Isso é bug ou é resultado?
#
# O script responde em três etapas, nesta ordem:
#
#   ETAPA 1 - O pipeline está quebrado?
#       Roda o MESMO caminho de predição sobre notícias do conjunto de teste,
#       cujo gabarito conhecemos. Se as falsas reais derem ~97% e as
#       verdadeiras ~2%, a função de predição está certa e o 0% é real.
#       Se as falsas reais também derem 0%, achamos o bug.
#
#   ETAPA 2 - O modelo hesita ou tem certeza?
#       Olha a probabilidade, não só o argmax. p(Falsa)=0,49 em todas seria
#       artefato de limiar; p(Falsa)=0,01 é um modelo confiantemente errado.
#
#   ETAPA 3 - É a lematização?
#       O corpo do FakeRecogna vem lematizado; as sintéticas são português
#       natural. Lematiza as 147 e reavalia.
#
# Roda em CPU. Não altera nenhum arquivo do projeto.

from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE = Path('/content/drive/MyDrive/TCC')

# ------------------------------------------------- 0. achar os arquivos
def listar_csvs():
    return sorted(BASE.rglob('*.csv'))


def procurar_csv(*nomes, mínimo_linhas=0, máximo_linhas=10**9, precisa_texto=True):
    """Procura por nome; se falhar, procura por conteúdo."""
    for nome in nomes:
        for caminho in listar_csvs():
            if caminho.name.lower() == nome.lower():
                return caminho
    # busca por conteúdo
    candidatos = []
    for caminho in listar_csvs():
        try:
            amostra = pd.read_csv(caminho, nrows=200)
            n = sum(1 for _ in open(caminho, encoding='utf-8', errors='ignore')) - 1
        except Exception:
            continue
        if not (mínimo_linhas <= n <= máximo_linhas):
            continue
        if precisa_texto:
            tem = any(amostra[c].astype(str).str.len().mean() > 100
                      for c in amostra.columns
                      if amostra[c].dtype == object)
            if not tem:
                continue
        candidatos.append((caminho, n, list(amostra.columns)))
    if len(candidatos) == 1:
        print('encontrado por conteúdo:', candidatos[0][0])
        return candidatos[0][0]
    print('\nNão consegui decidir sozinho. CSVs no Drive:')
    for caminho in listar_csvs():
        try:
            n = sum(1 for _ in open(caminho, encoding='utf-8', errors='ignore')) - 1
        except Exception:
            n = '?'
        marca = '  <-- candidato' if any(c[0] == caminho for c in candidatos) else ''
        print('   {:<58} {:>7} linhas{}'
              .format(str(caminho.relative_to(BASE)), n, marca))
    raise FileNotFoundError(
        'defina o caminho na mão: sint = pd.read_csv("<caminho acima>")')


def achar_modelo():
    for pasta in sorted(BASE.rglob('config.json')):
        if (pasta.parent / 'model.safetensors').exists() or \
           (pasta.parent / 'pytorch_model.bin').exists():
            return pasta.parent
    raise FileNotFoundError('não achei a pasta do modelo BERTimbau')


print('=' * 72)
print('0. INSUMOS')
print('=' * 72)

caminho_sint = procurar_csv('fake_news_sinteticas.csv', 'noticias_sinteticas.csv',
                            'sinteticas.csv',
                            mínimo_linhas=100, máximo_linhas=400)
sint = pd.read_csv(caminho_sint)
print('arquivo     :', caminho_sint.name)
if len(sint) != 147:
    print('  *** ATENÇÃO: {} linhas, mas a Fase 8 reportou 147. Existe um'
          .format(len(sint)))
    print('      fake_news_sinteticas_gemini35.csv com 204 linhas no Drive —')
    print('      confira se este é o mesmo conjunto que os notebooks usaram.')
col_texto = next((c for c in ('texto', 'noticia', 'notícia', 'conteudo', 'corpo')
                  if c in sint.columns), None)
if col_texto is None:  # pega a coluna de texto mais longa
    col_texto = max((c for c in sint.columns if sint[c].dtype == object),
                    key=lambda c: sint[c].astype(str).str.len().mean())
textos = sint[col_texto].fillna('').astype(str).tolist()
print('sintéticas  :', len(textos), 'notícias | coluna usada:', repr(col_texto))
print('  exemplo   :', ' '.join(textos[0].split())[:120], '...')

prep = pd.read_csv(procurar_csv('corpus_preparado.csv', 'corpus_base.csv',
                                mínimo_linhas=5000))
caminho = achar_modelo()
print('modelo      :', caminho)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(str(caminho))
mod = AutoModelForSequenceClassification.from_pretrained(str(caminho))
mod.eval()
disp = 'cuda' if torch.cuda.is_available() else 'cpu'
mod.to(disp)
print('dispositivo :', disp)
print('config id2label:', mod.config.id2label)


def probabilidades(lista, lote=16):
    """p(classe 1) para cada texto. Classe 1 = Falsa, por convenção do projeto."""
    saida = []
    for i in range(0, len(lista), lote):
        ent = tok(lista[i:i + lote], truncation=True, padding=True,
                  max_length=256, return_tensors='pt').to(disp)
        with torch.no_grad():
            logits = mod(**ent).logits
        saida.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(saida)


def relatorio(nome, probs, esperado=None):
    taxa = (probs > 0.5).mean()
    print('\n  {}'.format(nome))
    print('    classificadas como Falsa : {:.2%}  ({} de {})'
          .format(taxa, int((probs > 0.5).sum()), len(probs)))
    print('    p(Falsa) média / mediana : {:.4f} / {:.4f}'
          .format(probs.mean(), np.median(probs)))
    print('    p(Falsa) mínima / máxima : {:.4f} / {:.4f}'
          .format(probs.min(), probs.max()))
    if esperado is not None:
        ok = abs(taxa - esperado) < 0.15
        print('    esperado ~{:.0%}  ->  {}'
              .format(esperado, 'OK' if ok else '*** FORA DO ESPERADO ***'))
        return ok
    return None


# ============================================== ETAPA 1: o pipeline está são?
print('\n' + '=' * 72)
print('ETAPA 1 - O PIPELINE DE PREDIÇÃO ESTÁ ÍNTEGRO?')
print('=' * 72)
print('Mesmo caminho de código, agora sobre notícias de gabarito conhecido.')

col_txt_corpus = next((c for c in ('texto', 'texto_limpo', 'noticia', 'corpo')
                       if c in prep.columns), None)
if 'particao' in prep.columns:
    teste = prep[prep['particao'] == 'teste']
else:
    from sklearn.model_selection import train_test_split
    pos = np.arange(len(prep))
    _, pos_te = train_test_split(pos, test_size=0.2, random_state=42,
                                 stratify=prep['rotulo'].values)
    teste = prep.iloc[pos_te]

falsas = teste[teste['rotulo'] == 1][col_txt_corpus].fillna('').astype(str).tolist()[:150]
verdad = teste[teste['rotulo'] == 0][col_txt_corpus].fillna('').astype(str).tolist()[:150]

ok_f = relatorio('FALSAS reais do conjunto de teste (n={})'.format(len(falsas)),
                 probabilidades(falsas), esperado=0.97)
ok_v = relatorio('VERDADEIRAS reais do conjunto de teste (n={})'.format(len(verdad)),
                 probabilidades(verdad), esperado=0.03)

print('\n  ' + '-' * 68)
if ok_f and ok_v:
    print('  VEREDITO: pipeline íntegro. O modelo acerta o gabarito conhecido')
    print('  pelo mesmo caminho de código. O 0% nas sintéticas NÃO é bug de')
    print('  carregamento, de convenção de rótulo nem de função de predição.')
elif not ok_f and not ok_v:
    print('  VEREDITO: *** BUG ENCONTRADO ***  O modelo erra as duas classes')
    print('  conhecidas. Pode ser convenção de rótulo invertida ou modelo')
    print('  errado na pasta. Pare aqui e me mande esta saída.')
else:
    print('  VEREDITO: *** ALGO ERRADO ***  O modelo acerta uma classe e erra')
    print('  a outra. Isso indica saturação: ele prevê quase sempre a mesma')
    print('  classe. Pare aqui e me mande esta saída.')
print('  ' + '-' * 68)


# ================================= ETAPA 2: o modelo hesita ou tem certeza?
print('\n' + '=' * 72)
print('ETAPA 2 - AS SINTÉTICAS: HESITAÇÃO OU CERTEZA?')
print('=' * 72)

p_natural = probabilidades(textos)
relatorio('SINTÉTICAS - português natural (como está hoje)', p_natural)
print('\n    se baixássemos o limiar:')
for t in (0.40, 0.30, 0.20, 0.10, 0.05, 0.01):
    print('       p > {:.2f}  ->  {:>6.2%}'.format(t, (p_natural > t).mean()))

media = p_natural.mean()
print()
if media > 0.35:
    print('  O modelo HESITA: a probabilidade fica perto da fronteira. O 0% é')
    print('  em boa medida artefato do limiar de 0,50, e o artigo deve dizer')
    print('  isso em vez de reportar zero seco.')
else:
    print('  O modelo tem CERTEZA de que são verdadeiras. Não é artefato de')
    print('  limiar: ele está confiantemente errado, que é o resultado forte.')


# ====================================== ETAPA 3: é a forma superficial?
print('\n' + '=' * 72)
print('ETAPA 3 - É A LEMATIZAÇÃO?')
print('=' * 72)
print('O corpo do FakeRecogna vem lematizado ("umar", "segundar", "vezar").')
print('As sintéticas são português flexionado. O modelo pode estar falhando')
print('por não reconhecer a FORMA, e não por não enxergar a falsidade.')

try:
    import spacy
    try:
        nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
    except OSError:
        import subprocess, sys
        print('\nbaixando o modelo de língua do spaCy...')
        subprocess.run([sys.executable, '-m', 'spacy', 'download',
                        'pt_core_news_sm'], check=True)
        nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])

    lematizados = [' '.join(t.lemma_ for t in doc)
                   for doc in nlp.pipe(textos, batch_size=32)]

    print('\n  efeito da lematização:')
    print('    antes : ', ' '.join(textos[0].split())[:140])
    print('    depois: ', ' '.join(lematizados[0].split())[:140])
    print('\n  para comparar, um trecho do corpus de treino:')
    print('    corpus: ', ' '.join(falsas[0].split())[:140])

    p_lema = probabilidades(lematizados)
    relatorio('SINTÉTICAS - lematizadas', p_lema)

    sobe = (p_lema > 0.5).mean() - (p_natural > 0.5).mean()
    print('\n' + '=' * 72)
    print('VEREDITO FINAL')
    print('=' * 72)
    print('  detecção com texto natural    : {:.2%}'.format((p_natural > 0.5).mean()))
    print('  detecção com texto lematizado : {:.2%}'.format((p_lema > 0.5).mean()))
    print('  variação                      : {:+.2f} p.p.'.format(100 * sobe))
    print()
    if sobe > 0.05:
        print('  SUBIU. Boa parte da falha do BERTimbau é incompatibilidade de')
        print('  forma superficial: ele foi treinado em texto lematizado e não')
        print('  reconhece português flexionado. Isso é uma limitação do CORPUS,')
        print('  não uma propriedade do modelo, e muda o que a seção 4.5 afirma.')
    else:
        print('  NÃO SUBIU. Mesmo falando a mesma "língua" do treino, o modelo')
        print('  não vê nada de falso no texto sintético. O critério aprendido')
        print('  é que não corresponde a falsidade factual — que é exatamente')
        print('  a tese central do artigo, agora com o confundidor descartado.')

except ImportError:
    print('\nspaCy não está instalado. Rode numa célula separada e repita:')
    print('    !pip install -q spacy && !python -m spacy download pt_core_news_sm')

print('\n' + '=' * 72)
print('Me mande esta saída inteira.')
