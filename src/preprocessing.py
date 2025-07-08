import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sentence_transformers import SentenceTransformer

def parse_estimated_owners(x):
    try:
        if isinstance(x, str) and re.match(r"^\d[\d,]*\s*-\s*\d[\d,]*$", x.strip()):
            low, high = x.replace(",", "").split("-")
            return (int(low.strip()) + int(high.strip())) // 2
    except:
        pass
    return np.nan

def is_valid_owner_range(value):
    if isinstance(value, str):
        return bool(re.match(r"^\d+\s*-\s*\d+$", value.strip()))
    return False

def to_list_column(x):
    if pd.isna(x): return []
    return [i.strip() for i in x.split(",") if i.strip()]

def convert_to_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def embed_about_column(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    about_texts = df["About the game"].fillna("unknown").astype(str).tolist()
    embeddings = model.encode(about_texts, show_progress_bar=True)
    emb_df = pd.DataFrame(embeddings, columns=[f"About_emb_{i}" for i in range(embeddings.shape[1])])
    emb_df.index = df.index
    return pd.concat([df, emb_df], axis=1)

def preprocess(df, use_text_embedding=False):
    df = df.copy()

    # STEP 1: Bersihkan kolom Price
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df[df["Price"].notna()]
    df["Price"] = df["Price"].astype(float)

    # STEP 2: Estimated Owners
    df = df[df["Estimated owners"].apply(is_valid_owner_range)]
    df["EstimatedOwnersAvg"] = df["Estimated owners"].apply(parse_estimated_owners)

    # STEP 3: Numerik konversi
    numeric_cols = ["Peak CCU", "Required age", "Average playtime forever", "Median playtime forever"]
    df = convert_to_numeric(df, numeric_cols)
    df["Positive"] = pd.to_numeric(df["Positive"], errors="coerce").fillna(0).astype(int)
    df["Negative"] = pd.to_numeric(df["Negative"], errors="coerce").fillna(0).astype(int)

    # STEP 4: Tanggal rilis
    df["Release date"] = pd.to_datetime(df["Release date"], errors="coerce")
    df["release_year"] = df["Release date"].dt.year

    # STEP 5: Kolom list
    list_cols = ["Genres", "Tags", "Categories"]
    for col in list_cols:
        df[col] = df[col].apply(to_list_column).apply(lambda x: x if isinstance(x, list) else [])

    # STEP 6: Isi nilai kosong
    for col in ["About the game", "Developers", "Publishers"]:
        df[col] = df[col].fillna("unknown")
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # STEP 7: Drop baris tanpa target
    df = df[df["EstimatedOwnersAvg"].notna()]

    # STEP 8: Text Embedding opsional
    if use_text_embedding:
        df = embed_about_column(df)
    else:
        df.drop(columns=[col for col in df.columns if col.startswith("About_emb_")], errors="ignore", inplace=True)

    # STEP 9: Buang kolom tidak relevan
    drop_cols = [
        "AppID", "Name", "Estimated owners", "About the game", "Reviews",
        "Supported languages", "Full audio languages", "Windows", "Mac", "Linux",
        "User score", "Metacritic score", "Release date"
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df

def feature_engineering(df, use_embedding=False, top_n_developers=100):
    df = df.copy()

    # Multi-label: Tags, Genres, Categories
    mlb = lambda col: pd.DataFrame(
        MultiLabelBinarizer().fit_transform(df[col]),
        columns=[f"{col[:-1]}_{cls}" for cls in MultiLabelBinarizer().fit(df[col]).classes_],
        index=df.index
    )
    tags_df, genres_df, cat_df = mlb("Tags"), mlb("Genres"), mlb("Categories")

    # Developer encoding
    top_devs = df["Developers"].value_counts().nlargest(top_n_developers).index
    df["Dev_top"] = df["Developers"].apply(lambda x: x if x in top_devs else "Other")
    df["Dev_encoded"] = LabelEncoder().fit_transform(df["Dev_top"])

    # Numeric + Categorical
    numeric_cols = [
        "Price", "Peak CCU", "Required age", "Average playtime forever",
        "Median playtime forever", "Positive", "Negative", "release_year", "Dev_encoded"
    ]
    X_numeric = df[numeric_cols].reset_index(drop=True)
    X_cat = pd.concat([tags_df, genres_df, cat_df], axis=1).reset_index(drop=True)

    # Optional: Text Embedding
    embed_cols = [col for col in df.columns if col.startswith("About_emb_")]
    X_embed = df[embed_cols].reset_index(drop=True) if use_embedding and embed_cols else pd.DataFrame()

    # Final X
    X = pd.concat([X_numeric, X_cat, X_embed], axis=1)
    y = df["EstimatedOwnersAvg"]
    return X, y