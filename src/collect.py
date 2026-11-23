"""Collect recent YouTube videos across multiple categories via Data API v3."""
import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
API = 'https://www.googleapis.com/youtube/v3'
# IDs are the standard YouTube video categories (not search keywords).
CATEGORIES = {'music': '10', 'gaming': '20', 'sports': '17', 'education': '27', 'science_tech': '28', 'entertainment': '24'}


def api_get(session, endpoint, key, **params):
    response = session.get(f'{API}/{endpoint}', params={'key': key, **params}, timeout=30)
    if not response.ok:
        raise RuntimeError(f'YouTube API {response.status_code}: {response.text[:500]}')
    return response.json()


def duration_seconds(value):
    # ISO8601 durations, including optional days, hours, minutes and seconds.
    m = re.fullmatch(r'P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', value or '')
    if not m:
        return None
    d, h, minute, sec = [int(v or 0) for v in m.groups()]
    return d * 86400 + h * 3600 + minute * 60 + sec


def collect(key, per_category=100, region='BR', pause=0.15):
    rows = []
    with requests.Session() as session:
        for category, category_id in CATEGORIES.items():
            print(f'Coletando {category}...')
            page_token = None
            ids = []
            while len(ids) < per_category:
                params = dict(part='id', chart='mostPopular', regionCode=region,
                              videoCategoryId=category_id, maxResults=min(50, per_category - len(ids)))
                if page_token:
                    params['pageToken'] = page_token
                try:
                    payload = api_get(session, 'videos', key, **params)

                except RuntimeError as erro:
                    if "YouTube API 404" in str(erro):
                        print(
                            f"Categoria {category} (ID {category_id}) "
                            f"indisponível para a região {region}. Pulando..."
                        )
                        break

                    raise
                ids.extend(item['id'] for item in payload.get('items', []))
                page_token = payload.get('nextPageToken')
                if not page_token or not payload.get('items'):
                    break
                time.sleep(pause)
            for start in range(0, len(ids), 50):
                payload = api_get(session, 'videos', key, part='snippet,statistics,contentDetails',
                                  id=','.join(ids[start:start+50]), maxResults=50)
                for item in payload.get('items', []):
                    snippet = item.get('snippet', {})
                    stats = item.get('statistics', {})
                    details = item.get('contentDetails', {})
                    if 'viewCount' not in stats:
                        continue
                    rows.append({
                        'video_id': item['id'], 'category': category,
                        'category_id': snippet.get('categoryId', category_id),
                        'channel_id': snippet.get('channelId', ''),
                        'title': snippet.get('title', ''),
                        'published_at': snippet.get('publishedAt', ''),
                        'duration_seconds': duration_seconds(details.get('duration')),
                        'views': int(stats['viewCount']),
                        'likes': int(stats['likeCount']) if 'likeCount' in stats else None,
                        'comments': int(stats['commentCount']) if 'commentCount' in stats else None,
                    })
                time.sleep(pause)
    result = pd.DataFrame(rows).drop_duplicates('video_id')
    if result.empty:
        raise RuntimeError('Nenhum vídeo retornado. Verifique a chave e a região.')
    path = ROOT / 'data/raw/videos.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f'{len(result)} vídeos salvos em {path}')
    print('ATENÇÃO: amostra de vídeos populares; não representa vídeos comuns da plataforma.')


if __name__ == '__main__':
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-category', type=int, default=100)
    parser.add_argument('--region', default='BR')
    args = parser.parse_args()
    key = os.getenv('YOUTUBE_API_KEY')
    if not key:
        raise SystemExit('Defina YOUTUBE_API_KEY no arquivo .env')
    collect(key, args.per_category, args.region)
