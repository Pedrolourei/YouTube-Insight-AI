📊 YouTube Insight AI
Análise comparativa de vídeos do YouTube com rede neural MLP
   
Disciplina: Inteligência Artificial II — 2026/02
Curso: Sistemas de Informação — Faculdade Antonio Meneghetti (AMF)
Modalidade: Treinamento de rede neural própria
🎯 Problema
Por que alguns vídeos do YouTube apresentam desempenho superior ao de outros? Criadores de conteúdo precisam interpretar diversos fatores — título, duração, categoria, momento da publicação e engajamento — sem que a simples comparação de visualizações revele toda a história.

O YouTube Insight AI é um protótipo acadêmico que coleta metadados públicos, compara vídeos de diferentes categorias e treina uma rede neural MLP (Multilayer Perceptron) para classificar o desempenho relativo de vídeos. O objetivo é investigar padrões associados ao desempenho, não provar causalidade nem garantir viralização.
✨ Funcionalidades
- Coleta de metadados públicos pela YouTube Data API v3.
- Preparação e análise de vídeos de diferentes categorias.
- Extração de características de títulos, duração e publicação.
- Representação textual de títulos com TF-IDF na segunda versão do treinamento.
- Treinamento de uma rede neural MLP e comparação com regressão logística.
- Dashboard em Streamlit com gráficos, estatísticas e comparação de vídeos por URL.
- Exibição de pontuações experimentais da rede neural para categorias conhecidas.
- Avaliação por accuracy, F1-score, ROC-AUC, matriz de confusão e curva de perda.
🏗️ Arquitetura
YouTube Data API v3
        |
        v
src/collect.py  --> data/raw/videos.csv
        |
        v
src/prepare.py  --> data/processed/videos_features.csv
        |
        v
src/train.py    --> MLP + baseline de regressão logística
        |           --> models/ + outputs/
        v
app.py (Streamlit) --> Comparações, métricas e previsões experimentais
Organização do repositório
YouTube-Insight-AI/
├── app.py                    # Interface web
├── requirements.txt          # Dependências
├── README.md                 # Documentação
├── .env.example              # Exemplo de configuração (sem chave real)
├── src/
│   ├── __init__.py
│   ├── collect.py             # Coleta pela API
│   ├── prepare.py             # Engenharia de atributos e alvo
│   └── train.py               # Treinamento e avaliação
├── data/
│   ├── raw/                   # Dados coletados
│   └── processed/             # Dataset preparado
├── models/                    # Modelo e pré-processador salvos
└── outputs/
    ├── metrics.json           # Métricas do experimento
    └── figures/               # Gráficos gerados
Alguns arquivos de dados e modelos dependem da execução local. Confira o conteúdo das pastas na versão publicada.

🧠 Como funciona a IA
Dados e atributos
O modelo utiliza características disponíveis nos metadados dos vídeos, incluindo:
Grupo	Exemplos
Título	Número de caracteres, quantidade de palavras, presença de números e interrogação
Texto do título	Representação TF-IDF (V2)
Publicação	Hora e dia da semana (UTC)
Conteúdo	Duração do vídeo (transformação logarítmica)
Categoria	Categoria do vídeo, codificada para o modelo


Visualizações, curtidas e comentários são usados para análise do desempenho observado, mas não são entradas do modelo para prever o próprio sucesso.
Definição do alvo
A classificação binária above_expected compara as visualizações de cada vídeo com a mediana da amostra de vídeos populares da mesma categoria e faixa de idade. Grupos com menos de oito vídeos são descartados.
- 1: acima da mediana do grupo.
- 0: na mediana ou abaixo dela.
Importante: essa referência é relativa à amostra coletada de vídeos populares, e não ao conjunto de todos os vídeos do YouTube.
Treinamento
A rede neural é uma MLP supervisionada, implementada com TensorFlow/Keras. O experimento utiliza separação por canal para reduzir a presença de vídeos do mesmo canal em treino e teste, além de EarlyStopping para conter o sobreajuste. A V2 adiciona TF-IDF e compara o resultado com regressão logística.
📈 Resultados experimentais
O conjunto preparado utilizado no experimento contém 298 vídeos, divididos em 188 para treinamento, 47 para validação e 63 para teste.
Métrica (teste)	MLP V1	MLP V2
Accuracy	49,21%	53,97%
F1-score	0,610	0,525
ROC-AUC	0,492	0,565
Baseline da classe majoritária (accuracy)	52,38%	52,38%


Baseline de regressão logística (V2): accuracy de 47,62% e ROC-AUC de 0,476.

A V2 melhorou accuracy e ROC-AUC em relação à V1, mas piorou o F1-score. A accuracy da V2 supera a baseline majoritária por uma margem pequena. Durante o treinamento, a accuracy de treino chegou a aproximadamente 91%, enquanto a de validação permaneceu próxima de 53%, indicando overfitting.

Conclusão experimental: o sistema demonstra coleta, treinamento, inferência e visualização de uma rede neural própria, mas ainda não apresenta precisão suficiente para orientar decisões reais de publicação.
Exemplo de comparação no dashboard
Em um teste com dois vídeos da categoria entertainment:
Indicador	Demon Slayer	One Piece
Visualizações observadas	135.090	144.323
Curtidas observadas	5.489	3.966
Comentários observados	407	229
Curtidas por 100 visualizações	4,06	2,75
Saída da rede neural (experimental)	37,0%	45,6%


As porcentagens são saídas não calibradas do classificador, e não probabilidades reais de viralização. As estatísticas públicas podem mudar com o tempo.
🚀 Como executar localmente
1. Pré-requisitos
- Python 3.11 (64 bits recomendado).
- Acesso à internet para instalar dependências e consultar a API.
- Chave da YouTube Data API v3, criada no Google Cloud Console.
2. Obter o código
Baixe o ZIP do repositório ou clone:
git clone https://github.com/Pedrolourei/YouTube-Insight-AI.git
cd YouTube-Insight-AI
3. Criar o ambiente virtual
No PowerShell (Windows):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install streamlit
Se o PowerShell bloquear a ativação, use Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass e tente ativar novamente.
4. Configurar a chave da API
Crie um arquivo .env na raiz do projeto com:
YOUTUBE_API_KEY=COLOQUE_SUA_CHAVE_AQUI
Nunca publique .env ou chaves reais. Se houver .env.example, você pode copiá-lo e preencher somente o .env.
5. Coletar e preparar os dados
python -m src.collect --per-category 100 --region BR
python -m src.prepare
A coleta depende da disponibilidade das categorias e dos limites da API. Os totais podem variar; algumas categorias podem não estar disponíveis na região escolhida.
6. Treinar o modelo
python -m src.train
O script salva os artefatos do modelo, métricas e gráficos nas pastas models/ e outputs/. Caso já existam artefatos compatíveis, é possível abrir o dashboard sem treinar novamente.
7. Abrir o dashboard
python -m streamlit run app.py
Acesse http://localhost:8501 no navegador. Na aba Comparar URLs, informe links de vídeos públicos para consultar as estatísticas e, quando a categoria for conhecida pelo modelo, obter uma previsão experimental.
🔬 Como reproduzir e avaliar o experimento
1. Configure a API e colete vídeos públicos.
2. Execute a preparação para gerar atributos e o alvo binário.
3. Treine a rede neural e a baseline.
4. Confira outputs/metrics.json e os gráficos de avaliação.
5. Inicie o Streamlit e compare vídeos reais.
6. Observe as limitações: diferenças de categorias, tamanho da amostra, variação temporal das estatísticas e sobreajuste.
Os resultados não são necessariamente idênticos em uma nova coleta, porque os vídeos populares e seus contadores mudam ao longo do tempo.
⚠️ Limitações
- A amostra é pequena (298 vídeos) e concentrada em vídeos populares.
- A rede neural apresentou generalização limitada e sinais de overfitting.
- O alvo foi definido a partir de medianas da amostra completa antes da separação dos conjuntos, o que é uma limitação metodológica a corrigir em versões futuras.
- Vídeos de categorias ausentes do treinamento podem ser comparados estatisticamente, mas não recebem previsão da IA.
- As pontuações do modelo não são probabilidades calibradas.
- Correlação entre atributos e desempenho não prova causalidade.
- A API fornece contadores públicos acumulados; a coleta atual não constitui uma série histórica de desempenho desde a publicação.
🛠️ Próximos passos
- Ampliar e diversificar a base de vídeos e canais.
- Criar uma avaliação temporal e revisar a construção do alvo sem usar informações do conjunto de teste.
- Investigar características de thumbnails e contexto dos canais.
- Testar regularização, redução de dimensionalidade e calibração de probabilidades.
- Aplicar técnicas de interpretabilidade, como SHAP, com validação apropriada.
- Adicionar testes automatizados e melhorar a reprodutibilidade da coleta.
🔐 Segurança
Não inclua no repositório: .env, .venv/, __pycache__/, chaves de API ou outros segredos. Restrinja sua chave no Google Cloud à YouTube Data API v3 e substitua qualquer chave que tenha sido exposta.
📚 Tecnologias
Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · TF-IDF · Streamlit · Matplotlib · YouTube Data API v3
🎓 Contexto acadêmico
Projeto desenvolvido para o Trabalho 2 de Inteligência Artificial II (2026/02), curso de Sistemas de Informação, Faculdade Antonio Meneghetti (AMF). A solução se enquadra na Modalidade 1 — Treinamento de rede neural própria.
