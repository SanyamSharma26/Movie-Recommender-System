# app.py 

import os
import pickle
from typing import List, Tuple
import gdown
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- Config ----------
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", os.getenv("TMDB_API_KEY"))
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
HEADERS = {"User-Agent": "Mozilla/5.0"}
TOP_K = 5
REQ_TIMEOUT = 8

# Shared session with retries
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    ),
)

# ---------- CSS ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stApp { background: radial-gradient(1200px 600px at 10% 0%, #1e293b 0%, #0b1220 40%, #070b14 100%); color: #ecf0f1; }

    .app-header h1 {
        font-weight: 700; margin: 0.2rem 0 0.6rem 0;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #5f27cd);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 4px 18px rgba(255,255,255,0.08));
    }

    /* Tabs underline accent */
    .stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #ff6b6b, #48dbfb); height: 3px; }

    /* movie card */
    .movie-card {
        border-radius: 18px; background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(6px);
        overflow: hidden; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        box-shadow: 0 8px 22px rgba(0,0,0,.35); height: 100%;
    }
    .movie-card:hover { transform: translateY(-6px); box-shadow: 0 20px 40px rgba(0,0,0,.45); border-color: rgba(255,255,255,0.22); }

    .poster-wrap img { display: block; width: 100%; height: auto; border-bottom: 1px solid rgba(255,255,255,0.08); }

    .movie-title { font-weight: 600; font-size: 0.95rem; letter-spacing: .2px; margin: .65rem .9rem 0.35rem .9rem; color: #eef2f7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    .tmdb-link {
        display: inline-flex; align-items: center; gap: .35rem; margin: 0 .95rem .9rem .95rem; padding: .35rem .55rem;
        border-radius: 10px; font-size: 0.82rem; text-decoration: none !important; color: #a0b4ff;
        border: 1px dashed rgba(160,180,255,0.35); transition: all .15s ease;
    }
    .tmdb-link:hover { background: rgba(160,180,255,0.08); border-color: rgba(160,180,255,0.6); transform: translateY(-1px); }

    .footer { opacity: .65; font-size: .85rem; margin-top: 1.25rem; }
    /* --- Consistent button style --- */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #4f46e5, #3b82f6);
    color: white;
    border: none;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    transition: all .2s ease-in-out;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(0,0,0,.25);
}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data helpers ----------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_poster(movie_id: int) -> str | None:
    if not TMDB_API_KEY:
        return None
    try:
        r = SESSION.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=REQ_TIMEOUT,
        )
        r.raise_for_status()
        poster_path = r.json().get("poster_path")
        return f"{POSTER_BASE}{poster_path}" if poster_path else None
    except requests.RequestException:
        return None


def recommend(selected_title: str,
              movies_df: pd.DataFrame,
              similarity_matrix) -> Tuple[List[str], List[str | None], List[int]]:
    matches = movies_df.index[movies_df["title"] == selected_title]
    if len(matches) == 0:
        return [], [], []
    i_sel = int(matches[0])

    sims = list(enumerate(similarity_matrix[i_sel]))
    sims.sort(key=lambda x: x[1], reverse=True)
    top_idxs = [i for i, _ in sims if i != i_sel][:TOP_K]

    id_col = "movie_id" if "movie_id" in movies_df.columns else ("id" if "id" in movies_df.columns else None)

    names, posters, ids = [], [], []
    for i in top_idxs:
        row = movies_df.iloc[i]
        names.append(row["title"])
        if id_col is not None:
            tmdb_id = int(row[id_col])
            ids.append(tmdb_id)
            posters.append(fetch_poster(tmdb_id))
        else:
            ids.append(-1)
            posters.append(None)
    return names, posters, ids


# Correct TMDB movie-genre IDs 
GENRES = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,        # fixed
    "Horror": 27,
    "Musical": 10402,     # label change, still TMDB 'Music' id
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "Thriller": 53,
    "War": 10752,
}

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_by_genre(genre_id: int, page: int = 1) -> list[dict]:
    """Return a list of movie dicts (id, title, poster_path) for a genre."""
    if not TMDB_API_KEY:
        return []
    try:
        r = SESSION.get(
            "https://api.themoviedb.org/3/discover/movie",
            params={
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "include_adult": "false",
                "include_video": "false",
                "page": page,
                "with_genres": str(genre_id),
            },
            timeout=REQ_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json() or {}
        return data.get("results", [])
    except requests.RequestException:
        return []


# ---------- Load artifacts ----------

st.markdown('<div class="app-header"><h1>🎬 Movie Recommender System</h1></div>', unsafe_allow_html=True)


MOVIE_DICT_FILE_ID = "1FVfcM3cdNjtvlrjDwcgt0kOD7SaDvc4E"
SIMILARITY_FILE_ID = "12BZ1FILjFutgjxJvr9cWXyDPZ-MEjweA"

def ensure_from_drive(local_path: str, file_id: str, label: str):
    """Download local_path from Drive if missing."""
    if os.path.exists(local_path):
        return
    st.info(f"Downloading {label} from Google Drive…")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, local_path, quiet=False)

# get files if not present
ensure_from_drive("movie_dict.pkl", MOVIE_DICT_FILE_ID, "movie_dict.pkl")
ensure_from_drive("similarity.pkl", SIMILARITY_FILE_ID, "similarity.pkl")

# now load
try:
    movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open("similarity.pkl", "rb"))
except Exception as e:
    st.error(f"Failed to load data files: {e}")
    st.stop()


# ---------- UI ----------
tab1, tab2 = st.tabs(["🔎 Recommend by Movie", "🎭 Browse by Genre"])

with tab1:
    st.caption("Type or select a movie")
    selected = st.selectbox("", options=movies["title"].values, index=0, label_visibility="collapsed")
    if st.button("Recommend", type="primary"):
        names, posters, ids = recommend(selected, movies, similarity)
        if not names:
            st.warning("No recommendations found.")
        else:
            cols = st.columns(len(names))
            for col, name, poster, mid in zip(cols, names, posters, ids):
                with col:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    tmdb_url = f"https://www.themoviedb.org/movie/{mid}" if mid != -1 else None
                    if poster:
                        if tmdb_url:
                            st.markdown(
                                f'<a href="{tmdb_url}" target="_blank" class="poster-wrap"><img src="{poster}" alt="{name} poster"/></a>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(f'<div class="poster-wrap"><img src="{poster}" alt="{name} poster"/></div>',
                                        unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="poster-wrap"><img src="https://placehold.co/500x750/0b1220/9aa8d1?text=No+Poster" alt="No poster"/></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(f'<div class="movie-title">{name}</div>', unsafe_allow_html=True)
                    if tmdb_url:
                        st.markdown(f'<a class="tmdb-link" href="{tmdb_url}" target="_blank">View on TMDB ↗</a>',
                                    unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.caption("Pick a genre")
    genre_label = st.selectbox("", options=list(GENRES.keys()), index=list(GENRES.keys()).index("Action"),
                               label_visibility="collapsed")
    if st.button("Show Genre"):
        gid = GENRES[genre_label]
        # Try first two pages for broader coverage
        page1 = fetch_by_genre(gid, page=1)
        page2 = fetch_by_genre(gid, page=2)
        results = (page1 or []) + (page2 or [])

        if not results:
            st.warning("Couldn’t fetch movies for this genre right now.")
        else:
            # Take top 5 and render
            cards = results[:TOP_K]
            cols = st.columns(len(cards))
            for col, m in zip(cols, cards):
                with col:
                    mid = m.get("id", -1)
                    name = m.get("title") or m.get("name") or "Untitled"
                    poster_path = m.get("poster_path")
                    poster = f"{POSTER_BASE}{poster_path}" if poster_path else None
                    tmdb_url = f"https://www.themoviedb.org/movie/{mid}" if mid != -1 else None

                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    if poster:
                        if tmdb_url:
                            st.markdown(
                                f'<a href="{tmdb_url}" target="_blank" class="poster-wrap"><img src="{poster}" alt="{name} poster"/></a>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(f'<div class="poster-wrap"><img src="{poster}" alt="{name} poster"/></div>',
                                        unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="poster-wrap"><img src="https://placehold.co/500x750/0b1220/9aa8d1?text=No+Poster" alt="No poster"/></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(f'<div class="movie-title">{name}</div>', unsafe_allow_html=True)
                    if tmdb_url:
                        st.markdown(f'<a class="tmdb-link" href="{tmdb_url}" target="_blank">View on TMDB ↗</a>',
                                    unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Posters © TMDB • Personal API key required.</div>', unsafe_allow_html=True)