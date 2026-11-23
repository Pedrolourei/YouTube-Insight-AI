"""YouTube Insight AI: exploratory dashboard and experimental MLP predictions."""
import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
API = 'https://www.googleapis.com/youtube/v3/videos'
CATEGORY_IDS = {'10': 'music', '20': 'gaming', '17': 'sports', '27': 'education', '28': 'science_tech', '24': 'entertainment'}
FEATURES = ['title_length', 'title_words', 'has_number', 'has_question', 'published_hour_utc', 'published_weekday_utc', 'log_duration', 'category', 'title']

st.set_page_config(page_title='YouTube Insight AI', page_icon='📊', layout='wide')
st.title('📊 YouTube Insight AI')
st.caption('Comparação de vídeos de diferentes categorias e previsões experimentais de uma rede neural MLP.')
st.warning('Modelo exploratório: accuracy de teste ~54% na V2. Não interpreta causalidade e não deve orientar decisões comerciais isoladamente.')

@st.cache_data

def load_data():
    path = ROOT / 'data/processed/videos_features.csv'
    return pd.read_csv(path) if path.exists() else None

@st.cache_resource

def load_model():
    import tensorflow as tf
    model_path = ROOT / 'models/youtube_mlp.keras'
    prep_path = ROOT / 'models/preprocessor.joblib'
    if not model_path.exists() or not prep_path.exists():
        return None, None
    return tf.keras.models.load_model(model_path), joblib.load(prep_path)


def video_id_from_url(value):
    value = value.strip()
    if re.fullmatch(r'[A-Za-z0-9_-]{11}', value):
        return value
    parsed = urlparse(value)
    host = (parsed.hostname or '').lower()
    if host in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        parts = parsed.path.strip('/').split('/')
        if len(parts) == 2 and parts[0] in ('shorts', 'live', 'embed'):
            return parts[1]
    if host in ('youtu.be', 'www.youtu.be'):
        return parsed.path.strip('/').split('/')[0]
    return None


def iso_duration_seconds(value):
    m = re.fullmatch(r'P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', value or '')
    if not m:
        return np.nan
    d, h, minute, second = (int(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + minute * 60 + second


def fetch_video(video_id, key):
    response = requests.get(API, params={'part': 'snippet,statistics,contentDetails', 'id': video_id, 'key': key}, timeout=25)
    if not response.ok:
        raise RuntimeError(f'YouTube API HTTP {response.status_code}: {response.text[:250]}')
    items = response.json().get('items', [])
    if not items:
        raise ValueError('Vídeo não encontrado ou indisponível.')
    item = items[0]
    snippet, stats, details = item.get('snippet', {}), item.get('statistics', {}), item.get('contentDetails', {})
    published = pd.to_datetime(snippet.get('publishedAt'), utc=True, errors='coerce')
    return {'video_id': video_id, 'title': snippet.get('title', ''), 'category': CATEGORY_IDS.get(str(snippet.get('categoryId')), 'other'),
            'published_at': published, 'duration_seconds': iso_duration_seconds(details.get('duration')),
            'views': int(stats.get('viewCount', 0)), 'likes': int(stats['likeCount']) if 'likeCount' in stats else np.nan,
            'comments': int(stats['commentCount']) if 'commentCount' in stats else np.nan,
            'channel_id': snippet.get('channelId', '')}


def make_features(frame):
    x = frame.copy()
    x['title'] = x['title'].fillna('').astype(str)
    dates = pd.to_datetime(x['published_at'], utc=True, errors='coerce')
    x['title_length'] = x['title'].str.len()
    x['title_words'] = x['title'].str.split().str.len()
    x['has_number'] = x['title'].str.contains(r'\d').astype(int)
    x['has_question'] = x['title'].str.contains(r'\?').astype(int)
    x['published_hour_utc'] = dates.dt.hour
    x['published_weekday_utc'] = dates.dt.dayofweek
    x['log_duration'] = np.log1p(pd.to_numeric(x['duration_seconds'], errors='coerce').clip(lower=0))
    return x[FEATURES]


def predict(frame):
    model, preprocessor = load_model()
    if model is None:
        return None
    array = preprocessor.transform(make_features(frame))
    return model.predict(array, verbose=0).reshape(-1)


def show_comparison(frame):
    if frame.empty:
        st.info('Não há vídeos para comparar.')
        return
    columns = [c for c in ['title', 'category', 'views', 'likes', 'comments', 'duration_seconds'] if c in frame]
    st.dataframe(frame[columns], use_container_width=True, hide_index=True)
    if 'views' in frame:
        chart = frame[['title', 'views']].copy()
        chart['title'] = chart['title'].str.slice(0, 45)
        st.bar_chart(chart.set_index('title')['views'])
    if 'likes' in frame and 'views' in frame:
        ratios = frame[['title', 'likes', 'views']].copy()
        ratios['Curtidas por 100 visualizações'] = np.where(ratios['views'] > 0, 100 * ratios['likes'] / ratios['views'], np.nan)
        st.dataframe(ratios[['title', 'Curtidas por 100 visualizações']], hide_index=True, use_container_width=True)
    st.subheader('🧠 Análise da IA — previsões experimentais')
    model, preprocessor = load_model()
    if model is None or preprocessor is None:
        st.error('Modelo ou pré-processador não encontrado na pasta models/.')
        return
    # Não bloquear a análise inteira quando apenas um vídeo não pertence às categorias treinadas.
    known_categories = set()
    if hasattr(preprocessor, 'named_transformers_'):
        cat_transformer = preprocessor.named_transformers_.get('cat')
        if hasattr(cat_transformer, 'categories_'):
            known_categories = set(map(str, cat_transformer.categories_[0]))
    if not known_categories:
        st.error('Não foi possível identificar as categorias usadas no treinamento.')
        return
    valid = frame['category'].astype(str).isin(known_categories)
    result = frame[['title', 'category']].copy()
    result['Probabilidade estimada (%)'] = np.nan
    result['Situação'] = 'Categoria não vista no treinamento'
    if valid.any():
        try:
            probabilities = predict(frame.loc[valid])
            result.loc[valid, 'Probabilidade estimada (%)'] = np.round(probabilities * 100, 1)
            result.loc[valid, 'Situação'] = np.where(probabilities >= 0.5, 'Acima da referência (previsto)', 'Abaixo da referência (previsto)')
        except Exception as exc:
            st.error(f'Não foi possível gerar previsões: {exc}')
            return
    st.dataframe(result, use_container_width=True, hide_index=True)
    if valid.any():
        chart = result.loc[valid, ['title', 'Probabilidade estimada (%)']].copy()
        chart['title'] = chart['title'].str.slice(0, 48)
        st.bar_chart(chart.set_index('title')['Probabilidade estimada (%)'])
    if (~valid).any():
        st.warning('Alguns vídeos pertencem a categorias ausentes do treinamento. Eles continuam na comparação estatística, mas não recebem previsão da IA.')
    st.caption('A porcentagem é uma saída não calibrada do modelo, NÃO a chance de viralizar. A referência é a mediana de vídeos populares por categoria e faixa de idade; não demonstra causalidade. Accuracy de teste da V2: ~54%.')


tab1, tab2, tab3 = st.tabs(['📚 Dataset coletado', '🔗 Comparar URLs', '🧠 Avaliação do modelo'])
with tab1:
    df = load_data()
    if df is None:
        st.info('Dataset não encontrado. Execute: python -m src.collect e python -m src.prepare')
    else:
        st.metric('Vídeos preparados', len(df))
        categories = sorted(df['category'].dropna().unique())
        selected = st.multiselect('Categorias', categories, default=categories)
        filtered = df[df['category'].isin(selected)]
        st.subheader('Vídeos por categoria')
        st.bar_chart(filtered['category'].value_counts())
        st.caption('A amostra foi coletada de vídeos populares; não representa o YouTube inteiro.')
        choices = filtered.copy()
        choices['display'] = choices['title'].astype(str).str.slice(0, 75) + ' — ' + choices['video_id'].astype(str)
        selected_ids = st.multiselect('Selecione vídeos para comparar', choices['display'].tolist(), max_selections=8)
        if selected_ids:
            show_comparison(choices[choices['display'].isin(selected_ids)])
with tab2:
    st.write('Cole até 5 links de vídeos públicos, um por linha. Necessária a chave da API no arquivo .env.')
    urls = st.text_area('URLs', height=135, placeholder='https://www.youtube.com/watch?v=...')
    if st.button('Buscar e comparar vídeos', type='primary'):
        key = os.getenv('YOUTUBE_API_KEY')
        if not key:
            st.error('Configure YOUTUBE_API_KEY no arquivo .env.')
        else:
            ids = [video_id_from_url(s) for s in urls.splitlines() if s.strip()]
            if not ids or len(ids) > 5 or any(not i or not re.fullmatch(r'[A-Za-z0-9_-]{11}', i) for i in ids):
                st.error('Informe entre 1 e 5 URLs válidas ou IDs de 11 caracteres.')
            else:
                rows = []
                for video_id in dict.fromkeys(ids):
                    try:
                        rows.append(fetch_video(video_id, key))
                    except Exception as exc:
                        st.error(f'Erro no vídeo {video_id}: {exc}')
                if rows:
                    st.session_state['fetched_videos'] = pd.DataFrame(rows)
    if 'fetched_videos' in st.session_state:
        show_comparison(st.session_state['fetched_videos'])
with tab3:
    path = ROOT / 'outputs/metrics.json'
    if path.exists():
        metrics = json.loads(path.read_text(encoding='utf-8'))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Accuracy', f"{metrics.get('accuracy', 0):.1%}")
        c2.metric('F1', f"{metrics.get('f1', 0):.3f}")
        c3.metric('ROC-AUC', f"{metrics.get('roc_auc', 0):.3f}")
        c4.metric('Baseline majoritária', f"{metrics.get('majority_baseline_accuracy', 0):.1%}")
        if 'logistic_baseline_accuracy' in metrics:
            st.write(f"Regressão logística: accuracy {metrics['logistic_baseline_accuracy']:.1%} | ROC-AUC {metrics.get('logistic_baseline_roc_auc', 0):.3f}")
        for name, caption in [('loss.png', 'Curva de aprendizado'), ('confusion_matrix.png', 'Matriz de confusão')]:
            figure = ROOT / 'outputs/figures' / name
            if figure.exists():
                st.subheader(caption)
                st.image(str(figure))
        st.caption('Avaliação por canais separados. Resultados limitados pela amostra pequena e pelo alvo calculado antes da divisão de treino/teste.')
    else:
        st.info('Execute python -m src.train para gerar as métricas e os gráficos.')
