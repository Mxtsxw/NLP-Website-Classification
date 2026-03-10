"""
s3_client.py
------------
Centralised S3 access layer for the Streamlit app.

All functions are decorated with @st.cache_resource or @st.cache_data so each
artefact is downloaded at most once per app session, stored in memory, and
never re-fetched on reruns.

Credentials are read from Streamlit secrets (secrets.toml):
"""

from __future__ import annotations

import io
import json
import logging
import pickle
from typing import Any

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError
from scipy.sparse import load_npz

log = logging.getLogger(__name__)


# ─── Credential helpers ───────────────────────────────────────────────────────

def _get_aws_cfg() -> dict:
    """Read AWS config from st.secrets, fall back to boto3 default chain."""
    try:
        cfg = dict(st.secrets.get("aws", {}))
    except Exception:
        cfg = {}
    return cfg


@st.cache_resource(show_spinner=False)
def _get_s3() -> boto3.client:
    """Return a cached boto3 S3 client."""
    cfg = _get_aws_cfg()
    kwargs: dict = {}

    if cfg.get("access_key_id"):
        kwargs["aws_access_key_id"]     = cfg["access_key_id"]
        kwargs["aws_secret_access_key"] = cfg["secret_access_key"]
    if cfg.get("region"):
        kwargs["region_name"] = cfg["region"]

    return boto3.client("s3", **kwargs)


def _bucket_and_prefix() -> tuple[str, str]:
    cfg    = _get_aws_cfg()
    bucket = cfg.get("bucket", "")
    prefix = cfg.get("prefix", "").rstrip("/")
    if not bucket:
        raise ValueError(
            "S3 bucket not configured. Add [aws] bucket = '...' to .streamlit/secrets.toml"
        )
    return bucket, prefix


def _s3_key(path: str) -> str:
    """Build a full S3 key from a relative artefact path."""
    _, prefix = _bucket_and_prefix()
    return f"{prefix}/{path}" if prefix else path


# ─── Raw download helper ──────────────────────────────────────────────────────

def _download_bytes(s3_path: str) -> bytes:
    """Download an S3 object and return raw bytes."""
    s3                = _get_s3()
    bucket, _         = _bucket_and_prefix()
    key               = _s3_key(s3_path)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Failed to download s3://{bucket}/{key}: {e}") from e


# ─── Typed loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models from S3...")
def load_pickle(s3_path: str) -> Any:
    """Download and unpickle an artefact (model, vectorizer, encoder)."""
    log.info(f"S3 download (pickle): {s3_path}")
    return pickle.loads(_download_bytes(s3_path))


@st.cache_data(show_spinner="Loading arrays from S3...")
def load_npy(s3_path: str) -> np.ndarray:
    """Download a .npy file and return a numpy array."""
    log.info(f"S3 download (npy): {s3_path}")
    return np.load(io.BytesIO(_download_bytes(s3_path)), allow_pickle=False)


@st.cache_data(show_spinner="Loading sparse matrix from S3...")
def load_npz_matrix(s3_path: str):
    """Download a .npz file and return a scipy sparse matrix."""
    log.info(f"S3 download (npz): {s3_path}")
    return load_npz(io.BytesIO(_download_bytes(s3_path)))


@st.cache_data(show_spinner="Loading dataframe from S3...")
def load_parquet(s3_path: str) -> pd.DataFrame:
    """Download a .parquet file and return a DataFrame."""
    log.info(f"S3 download (parquet): {s3_path}")
    return pd.read_parquet(io.BytesIO(_download_bytes(s3_path)))


@st.cache_data(show_spinner="Loading CSV from S3...")
def load_csv(s3_path: str) -> pd.DataFrame:
    """Download a .csv file and return a DataFrame."""
    log.info(f"S3 download (csv): {s3_path}")
    return pd.read_csv(io.BytesIO(_download_bytes(s3_path)))


@st.cache_data(show_spinner="Loading results from S3...")
def load_json(s3_path: str) -> dict:
    """Download a .json file and return a dict."""
    log.info(f"S3 download (json): {s3_path}")
    return json.loads(_download_bytes(s3_path).decode("utf-8"))


# ─── Convenience bundle ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading all artefacts from S3...")
def load_all_artefacts() -> dict:
    """
    Download and return every artefact the Streamlit app needs in one call.
    This function is cached as a resource so it runs exactly once per session.

    Returns a dict with keys:
        tfidf, svd, le, knn, svm          (sklearn objects)
        X_train_knn, X_test_knn           (dense numpy arrays)
        X_train_tfidf, X_test_tfidf       (scipy sparse matrices)
        y_train, y_test                   (numpy arrays)
        train_features, test_features     (DataFrames, no text column)
        train_texts, test_texts           (DataFrames with text column)
        model_results                     (dict with accuracy, reports, CMs)
    """
    return {
        # Transformers + models
        "tfidf":           load_pickle("models/tfidf.pkl"),
        "svd":             load_pickle("models/svd150.pkl"),
        "le":              load_pickle("models/label_encoder.pkl"),
        "knn":             load_pickle("models/knn.pkl"),
        "svm":             load_pickle("models/svm.pkl"),
        # Feature matrices
        "X_train_knn":     load_npy("data/X_train_knn.npy"),
        "X_test_knn":      load_npy("data/X_test_knn.npy"),
        "X_train_tfidf":   load_npz_matrix("data/X_train_tfidf.npz"),
        "X_test_tfidf":    load_npz_matrix("data/X_test_tfidf.npz"),
        "y_train":         load_npy("data/y_train.npy"),
        "y_test":          load_npy("data/y_test.npy"),
        # Dataframes
        "train_features":  load_csv("data/train_features.csv"),
        "test_features":   load_csv("data/test_features.csv"),
        "train_texts":     load_parquet("data/train_texts.parquet"),
        "test_texts":      load_parquet("data/test_texts.parquet"),
        # Evaluation summary
        "model_results":   load_json("data/model_results.json"),
    }
