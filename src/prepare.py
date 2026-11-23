"""Feature engineering and within-category age-adjusted target construction."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    df = pd.read_csv(ROOT / 'data/raw/videos.csv')
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce')
    # Use the current collection instant, not a fixed date.
    collected_at = pd.Timestamp.now(tz='UTC')
    df['age_days'] = (collected_at - df['published_at']).dt.total_seconds() / 86400
    df['title'] = df['title'].fillna('')
    df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce')
    df['views'] = pd.to_numeric(df['views'], errors='coerce')
    df = df.dropna(subset=['published_at', 'age_days', 'duration_seconds', 'views', 'category', 'channel_id'])
    df = df[(df['age_days'] >= 1) & (df['duration_seconds'] >= 15) & (df['views'] >= 0)].copy()
    df['title_length'] = df['title'].str.len()
    df['title_words'] = df['title'].str.split().str.len()
    df['has_number'] = df['title'].str.contains(r'\d').astype(int)
    df['has_question'] = df['title'].str.contains(r'\?').astype(int)
    df['published_hour_utc'] = df['published_at'].dt.hour
    df['published_weekday_utc'] = df['published_at'].dt.dayofweek
    df['log_duration'] = np.log1p(df['duration_seconds'])
    # Target is relative to videos in the same category and age bin.
    # Avoid using the target (views, likes, comments) as input features.
    df['age_bin'] = pd.cut(df['age_days'], bins=[1, 3, 7, 14, 30, 90, 365, float('inf')],
                           labels=False, include_lowest=True)
    df['log_views'] = np.log1p(df['views'])
    grp = df.groupby(['category', 'age_bin'], observed=True)['log_views']
    df['group_size'] = grp.transform('size')
    df['group_median'] = grp.transform('median')
    df = df[df['group_size'] >= 8].copy()
    if len(df) < 80:
        raise ValueError(f'Apenas {len(df)} exemplos com grupos suficientes; amplie a coleta.')
    df['above_expected'] = (df['log_views'] > df['group_median']).astype(int)
    path = ROOT / 'data/processed/videos_features.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f'Dataset preparado: {len(df)} vídeos, classe positiva={df.above_expected.mean():.1%}')
    print('ATENÇÃO: alvo relativo à amostra popular, não ao YouTube inteiro.')


if __name__ == '__main__':
    prepare()
