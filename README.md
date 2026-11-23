# YouTube Insight AI

Projeto acadêmico de IA II (modalidade 1): rede neural MLP própria para identificar padrões associados ao desempenho relativo de vídeos públicos de diferentes categorias no YouTube.

## Pergunta de pesquisa

Características disponíveis na publicação (título, duração, horário e categoria) ajudam a distinguir vídeos com visualizações acima ou abaixo da mediana de vídeos da mesma categoria e faixa de idade, **dentro da amostra coletada**?

## Arquitetura

YouTube Data API v3 -> CSV bruto -> engenharia de atributos e rótulos -> pré-processamento (ajustado apenas no treino) -> MLP Keras -> avaliação em canais não vistos -> gráficos e métricas.

## Requisitos

- Python 3.11 recomendado, computador com acesso à internet e uma chave da YouTube Data API v3.
- Crie um projeto no Google Cloud, habilite **YouTube Data API v3**, crie uma chave de API e configure as restrições adequadas.
- Nunca publique a chave nem o arquivo `.env` no GitHub.

## Execução (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
# Coloque a chave de API no .env, salve e feche.
python -m src.collect --per-category 100 --region BR
python -m src.prepare
python -m src.train
```

## Execução (Linux/macOS)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edite .env com a chave da API
python -m src.collect --per-category 100 --region BR
python -m src.prepare
python -m src.train
```

## Dados e variáveis

Categorias: música, games, esportes, educação, ciência/tecnologia e entretenimento. A coleta usa o endpoint `videos.list` com `chart=mostPopular` por categoria e região BR. Cada vídeo possui ID, canal, título, data, duração, visualizações, curtidas e comentários quando disponíveis.

A rede recebe somente: tamanho do título, número de palavras, presença de número ou interrogação, hora UTC, dia da semana UTC, log da duração e categoria. **Não** recebe visualizações, curtidas ou comentários como entradas.

O alvo binário é `log1p(visualizações)` maior que a mediana do mesmo grupo `(categoria, faixa de idade)` dentro da amostra. Grupos com menos de oito vídeos são descartados. A idade é calculada na coleta/preparação; o alvo reflete um instantâneo e não desempenho futuro.

## Rede neural e avaliação

MLP com duas camadas densas de 64 e 32 neurônios (ReLU), dropout de 25%, saída sigmoide, Adam e binary cross-entropy. Treino/validação/teste são separados por **canal**, para reduzir a repetição de criadores nas partições. O pré-processamento é ajustado apenas no treino. Early stopping acompanha a perda de validação.

Arquivos produzidos:

- `data/raw/videos.csv`: dados coletados
- `data/processed/videos_features.csv`: atributos e rótulos
- `models/youtube_mlp.keras`: rede treinada
- `models/preprocessor.joblib`: transformações de entrada
- `outputs/metrics.json`: accuracy, F1, ROC AUC e baseline majoritário
- `outputs/figures/loss.png`: curvas de treino/validação
- `outputs/figures/confusion_matrix.png`: matriz de confusão

**Não há resultados numéricos reais incluídos:** execute a coleta e o treino para gerá-los. Não invente métricas na apresentação.

## Limitações metodológicas importantes

1. `mostPopular` cria **viés de seleção severo**: são vídeos já populares, não uma amostra representativa de todos os vídeos. O modelo não pode afirmar que prevê sucesso no YouTube em geral.
2. Não existe inferência causal: a rede encontra associações, não demonstra **por que** um vídeo fez sucesso.
3. O endpoint fornece contadores acumulados, não séries históricas; comparações por idade são aproximações imperfeitas.
4. O alvo usa medianas da amostra inteira, inclusive teste, como referência descritiva. Para avaliação de produção, congele referências usando apenas dados anteriores ao teste e faça validação temporal.
5. Canal, orçamento de divulgação, tamanho de público, thumbnails, tendência do assunto e tráfego externo podem confundir resultados.
6. A análise de importância de atributos e o dashboard ainda não estão implementados nesta primeira etapa.
7. O número de vídeos e as categorias disponíveis podem variar; aumente a coleta se grupos ou classes forem insuficientes.

## Próximas etapas

- Dashboard Streamlit para comparação de vídeos e métricas.
- Permutation importance ou SHAP para explicações com cautela.
- Amostragem mais representativa por buscas variadas e diferentes canais, sem depender apenas de `mostPopular`.
- Referência temporal e análise de vídeos de um mesmo canal/nicho para controlar confundidores.

## Referências

- https://developers.google.com/youtube/v3/docs/videos/list
- https://www.tensorflow.org/guide/keras
- https://scikit-learn.org/stable/modules/cross_validation.html
