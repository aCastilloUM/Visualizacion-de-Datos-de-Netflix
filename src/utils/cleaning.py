# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import difflib
from typing import Dict, List
import pandas as pd
import collections
import numpy as np


# ---------------- Países ----------------
COUNTRY_ALIASES: Dict[str, str] = {
    "Usa": "United States","U.s.": "United States","U.s.a.": "United States",
    "United States Of America": "United States",
    "England": "United Kingdom","Uk": "United Kingdom","U.k.": "United Kingdom",
    "Great Britain": "United Kingdom","Britain": "United Kingdom",
    "Korea, South": "South Korea","Republic Of Korea": "South Korea","Korea": "South Korea",
    "Viet Nam": "Vietnam","Russian Federation": "Russia","Uae": "United Arab Emirates",
    "Iran, Islamic Republic Of": "Iran","Hong Kong S.a.r.": "Hong Kong",
    "Taiwan, Province Of China": "Taiwan","Macau, Sar China": "Macau",
    "Czechia": "Czech Republic","Moldova, Republic Of": "Moldova",
    "Tanzania, United Republic Of": "Tanzania",
    "Bolivia, Plurinational State Of": "Bolivia",
    "Venezuela, Bolivarian Republic Of": "Venezuela",
    "Syrian Arab Republic": "Syria",
    "Estados Unidos": "United States","Reino Unido": "United Kingdom",
    "Corea Del Sur": "South Korea","Emiratos Arabes Unidos": "United Arab Emirates",
    "Republica Checa": "Czech Republic",
}

CANON_COUNTRIES: List[str] = [
    "United States","United Kingdom","India","Japan","South Korea","Canada","France","Spain","Germany",
    "Italy","Mexico","Turkey","Brazil","Australia","Argentina","Colombia","Chile","Peru","Russia",
    "China","Hong Kong","Taiwan","Thailand","Philippines","Indonesia","Malaysia","Singapore","Vietnam",
    "United Arab Emirates","Saudi Arabia","Egypt","South Africa","Nigeria","Kenya","Ghana","Morocco",
    "Netherlands","Belgium","Sweden","Norway","Denmark","Finland","Poland","Czech Republic","Austria",
    "Switzerland","Portugal","Ireland","Greece","Israel","Iran","Iraq","Syria","Lebanon","Jordan",
    "New Zealand","Bangladesh","Pakistan","Sri Lanka","Nepal","Romania","Hungary","Bulgaria","Serbia",
    "Croatia","Slovakia","Slovenia","Ukraine","Belarus","Lithuania","Latvia","Estonia","Iceland",
    "Luxembourg","Uruguay","Paraguay","Bolivia","Ecuador","Venezuela","Guatemala","Costa Rica",
    "Panama","El Salvador","Honduras","Nicaragua","Dominican Republic","Cuba","Haiti","Jamaica",
    "Qatar","Kuwait","Bahrain","Oman","Yemen","Algeria","Tunisia","Libya","Ethiopia","Tanzania",
    "Uganda","Zimbabwe","Zambia","Botswana","Namibia","Cameroon","Ivory Coast","Senegal","Mali",
    "Armenia","Azerbaijan","Georgia","Kazakhstan","Uzbekistan","Kyrgyzstan","Mongolia",
    "Macau","Malta","Cyprus","Liechtenstein","Andorra","Monaco","San Marino","Vatican City"
]

# ---------------- listed_in to género canónico (ES) ----------------
LISTED_IN_TO_CANON_GENRE = {
    # Cine/TV por género principal
    "Comedies": "Comedia",
    "TV Comedies": "Comedia",
    "Dramas": "Drama",
    "TV Dramas": "Drama",
    "Horror Movies": "Terror",
    "TV Horror": "Terror",
    "Thrillers": "Thriller",
    "TV Thrillers": "Thriller",
    "Crime TV Shows": "Crimen/Misterio",
    "TV Mysteries": "Crimen/Misterio",

    # Acción / Aventura
    "Action & Adventure": "Acción/Aventura",
    "TV Action & Adventure": "Acción/Aventura",

    # Sci-Fi & Fantasy
    "Sci-Fi & Fantasy": "Ciencia Ficción/Fantasía",
    "TV Sci-Fi & Fantasy": "Ciencia Ficción/Fantasía",

    # Romance
    "Romantic Movies": "Romance",
    "Romantic TV Shows": "Romance",

    # Documental / Docuseries / Ciencia & Naturaleza
    "Documentaries": "Documental",
    "Docuseries": "Documental",
    "Science & Nature TV": "Ciencia/Naturaleza",

    # Infantil / Familiar
    "Children & Family Movies": "Infantil/Familiar",
    "Kids' TV": "Infantil/Familiar",
    "Teen TV Shows": "Infantil/Familiar",

    # Anime
    "Anime Features": "Anime",
    "Anime Series": "Anime",

    # Música / Deportes / Reality
    "Music & Musicals": "Música",
    "Sports Movies": "Deportes",
    "Reality TV": "Reality",

    # Stand-Up
    "Stand-Up Comedy": "Stand-Up",
    "Stand-Up Comedy & Talk Shows": "Stand-Up",

    # Internacional / Regional
    "International Movies": "Internacional/Regional",
    "International TV Shows": "Internacional/Regional",
    "British TV Shows": "Internacional/Regional",
    "Korean TV Shows": "Internacional/Regional",
    "Spanish-Language TV Shows": "Internacional/Regional",

    # Clásicos / Culto / Independiente
    "Classic Movies": "Clásicos/Culto",
    "Cult Movies": "Clásicos/Culto",
    "Classic & Cult TV": "Clásicos/Culto",
    "Independent Movies": "Independiente",

    # Fe / Espiritualidad
    "Faith & Spirituality": "Fe/Espiritualidad",

    "TV Shows": "TV (General)",  
}

# ---------------- Text Mining (EN) ----------------

ENGLISH_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","for","to","from","of","in","on","at","by","with","as","is","are","was","were","be","been","being",
    "this","that","these","those","it","its","they","them","their","there","here","you","your","we","our","us","he","she","his","her","i","me","my",
    "not","no","yes","do","does","did","doing","done","can","could","should","would","may","might","will","just","than","so","such","only","very","more","most","much","many",
    "into","over","under","between","through","about","across","after","before","again","once","also","all","any","both","each","few","other","some","own","same","too","up","down","out","off",
    "film","movie","series","season","seasons","episode","episodes","show","netflix",
}


def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _clean_country_token(tok: str) -> str:
    t = str(tok).strip()
    t = re.sub(r"\s+", " ", t).replace("'", "")
    t = _strip_accents(t).title()
    t = re.sub(r"\s*\(.*?\)\s*$", "", t).strip()
    return t

def _canon_country(tok: str) -> str:
    if tok in CANON_COUNTRIES:
        return tok
    cand = difflib.get_close_matches(tok, CANON_COUNTRIES, n=1, cutoff=0.85)
    return cand[0] if cand else tok

def expand_and_normalize_countries(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    dfx["country"] = dfx["country"].fillna("").astype(str)
    dfx["country_tokens"] = dfx["country"].str.split(",")
    dfx = dfx.explode("country_tokens", ignore_index=True)
    dfx["country_final"] = (
        dfx["country_tokens"]
        .map(lambda x: _clean_country_token(x) if isinstance(x, str) else "")
        .map(lambda x: COUNTRY_ALIASES.get(x, x))
        .map(_canon_country)
    )
    dfx = dfx[dfx["country_final"] != ""].copy()
    return dfx

# ---------------- Ratings ----------------
RATING_ALIASES = {
    "TV MA": "TV-MA","TV 14": "TV-14","TV PG": "TV-PG","TV G": "TV-G","TV Y": "TV-Y","TV Y7": "TV-Y7",
    "TV Y7 FV": "TV-Y7-FV","PG 13": "PG-13","NC 17": "NC-17","UR": "NR","NR.": "NR"
}

def normalize_and_explode_ratings(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    dfx = dfx.dropna(subset=["rating"])
    dfx["rating_tokens"] = dfx["rating"].astype(str).str.split(",")
    dfx = dfx.explode("rating_tokens", ignore_index=True)

    def _norm(tok: str) -> str:
        t = str(tok).strip()
        t = re.sub(r"\s+", " ", t).upper().replace(".", "")
        t = t.replace(" TV ", " TV-").replace(" Y7 FV", "-Y7-FV").replace(" Y7", "-Y7")
        t = RATING_ALIASES.get(t, t)
        t = (t.replace("TV MA", "TV-MA").replace("TV 14", "TV-14").replace("TV PG", "TV-PG")
               .replace("TV G", "TV-G").replace("TV Y", "TV-Y").replace("TV Y7", "TV-Y7")
               .replace("PG 13", "PG-13").replace("NC 17", "NC-17"))
        t = re.sub(r"-{2,}", "-", t)
        return t

    dfx["rating_norm"] = dfx["rating_tokens"].map(_norm)
    dfx = dfx[dfx["rating_norm"] != ""].copy()
    return dfx

# ---------------- Fechas ----------------
def ensure_datetime(df: pd.DataFrame, col: str = "date_added") -> pd.DataFrame:
    dfx = df.copy()
    dfx[col] = pd.to_datetime(dfx[col], errors="coerce")
    return dfx

def add_year_and_month(df: pd.DataFrame, date_col: str = "date_added") -> pd.DataFrame:
    dfx = ensure_datetime(df, date_col)
    dfx["year_added"] = dfx[date_col].dt.year
    dfx["month_added"] = dfx[date_col].dt.month
    return dfx

# ---------------- listed_in ----------------
def explode_listed_in(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    dfx = dfx.dropna(subset=["listed_in"])
    dfx["listed_in"] = dfx["listed_in"].astype(str).str.split(",")
    dfx = dfx.explode("listed_in", ignore_index=True)
    dfx["listed_in"] = dfx["listed_in"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.title()
    dfx = dfx[dfx["listed_in"] != ""]
    return dfx

# ---------------- Directores ----------------
def expand_and_normalize_directors(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    dfx["director"] = dfx["director"].fillna("").astype(str)
    dfx["director_tokens"] = dfx["director"].str.split(",")
    dfx = dfx.explode("director_tokens", ignore_index=True)
    dfx["director_final"] = (
        dfx["director_tokens"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    dfx = dfx[dfx["director_final"] != ""].copy()
    return dfx

def map_listed_in_to_genre_token(token: str) -> str:
    if not isinstance(token, str):
        return ""
    t = token.strip()
    if not t:
        return ""
    return LISTED_IN_TO_CANON_GENRE.get(t, t)

def add_genre_from_listed_in(df: pd.DataFrame, listed_col: str = "listed_in") -> pd.DataFrame:
    dfx = df.copy()
    dfx["genre_main"] = dfx[listed_col].map(map_listed_in_to_genre_token)
    dfx = dfx[dfx["genre_main"].astype(str).str.strip() != ""]
    return dfx

# ---------------- Map ratings a audiencias ----------------
def map_rating_to_audience(rating: str, mode: str = "adult_kids") -> str:
    if not isinstance(rating, str) or rating.strip() == "":
        return "Adulto"

    r = rating.strip().upper()

    infantiles = {"G", "TV-Y", "TV-Y7", "TV-G", "PG", "TV-PG", "TV-Y7-FV", "PG-13", "TV-14"}
    adultos     = {"R", "NC-17", "TV-MA"}

    if r in infantiles:
        return "Infantil"
    elif r in adultos:
        return "Adulto"
    else:
        # Ratings como PG-13, NR, son tratados como adultos por defecto
        return "Adulto"

# ---------------- Elenco / Actores ----------------
def expand_and_normalize_cast(df: pd.DataFrame) -> pd.DataFrame:

    dfx = df.copy()
    dfx["cast"] = dfx["cast"].fillna("").astype(str)
    dfx["cast_tokens"] = dfx["cast"].str.split(",")
    dfx = dfx.explode("cast_tokens", ignore_index=True)
    dfx["cast_final"] = (
        dfx["cast_tokens"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    dfx = dfx[dfx["cast_final"] != ""].copy()
    return dfx

# ---------------- Duraciones ----------------
def normalize_duration(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    def _norm_type(x: str) -> str:
        s = str(x).strip().lower().replace("-", " ")
        s = re.sub(r"\s+", " ", s)
        if s in {"movie", "pelicula", "películas"}:
            return "Movie"
        if s in {"tv show", "tv shows", "serie", "series"}:
            return "TV Show"
        return str(x).strip() 

    dfx["type"] = dfx["type"].map(_norm_type)

    # Normalizar duration a str lowercase
    dur = dfx["duration"].fillna("").astype(str).str.strip()
    dur_low = dur.str.lower()

    # Columnas de salida
    dfx["duration_minutes"] = np.nan
    dfx["duration_seasons"] = np.nan

 
    is_min = dur_low.str.contains(r"\bmin\b")
    dfx.loc[is_min, "duration_minutes"] = (
        dur_low[is_min].str.extract(r"(\d+)\s*min", expand=False).astype(float)
    )

    is_season = dur_low.str.contains(r"season")
    dfx.loc[is_season, "duration_seasons"] = (
        dur_low[is_season].str.extract(r"(\d+)\s*season", expand=False).astype(float)
    )

    return dfx


_WORD_RE = re.compile(r"[a-z]+")

# Minusculas, solo letras a-z, split simple 
def normalize_to_words_en(text: str, min_len: int = 3) -> list[str]:
    
    if not isinstance(text, str) or not text:
        return []
    s = text.lower()
    tokens = _WORD_RE.findall(s)
    out = [t for t in tokens if len(t) >= min_len and t not in ENGLISH_STOPWORDS]
    return out

def count_top_words(series: pd.Series, topn: int = 20, min_len: int = 3) -> pd.Series:
    counter = collections.Counter()
    for txt in series.dropna().astype(str):
        counter.update(normalize_to_words_en(txt, min_len=min_len))
    if not counter:
        return pd.Series(dtype=int)
    most_common = counter.most_common(topn)
    s = pd.Series({w: c for w, c in most_common}, dtype=int)
    return s.sort_values(ascending=True)

