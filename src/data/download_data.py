"""
DownLoad FNSPID dataset from Huggingface
"""

import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


def download_with_resume(
    url: str, out_path: str, chunk_size=1024 * 1024, max_retries=8
):
    # requests session + retry
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=1.0,  # 1s, 2s, 4s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    downloaded = os.path.getsize(out_path) if os.path.exists(out_path) else 0

    head = session.head(url, allow_redirects=True, timeout=30)
    total = int(head.headers.get("Content-Length", 0))

    headers = {
        "Accept-Encoding": "identity",
    }
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    # GET streaming
    with session.get(url, stream=True, headers=headers, timeout=(30, 60)) as r:
        r.raise_for_status()

        if "Content-Range" in r.headers:
            total = int(r.headers["Content-Range"].split("/")[-1])

        mode = "ab" if downloaded > 0 else "wb"
        pbar = tqdm(
            total=total,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(out_path),
        )

        with open(out_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        pbar.close()

    return out_path


def extrace_data_set_by_symbols(
    symbol_list: list,
    news_db_path: str,
    output_path: str,
):
    writer = None
    for _, news_chunk in tqdm(enumerate(pd.read_csv(news_db_path, chunksize=1000000))):
        filtered_news_chunk = news_chunk[news_chunk["Stock_symbol"].isin(symbol_list)]
        filtered_news_chunk["Publisher"] = filtered_news_chunk["Publisher"].astype(str)
        filtered_news_chunk["Date"] = filtered_news_chunk["Date"].astype(str)
        filtered_news_chunk["Article_title"] = filtered_news_chunk[
            "Article_title"
        ].astype(str)
        filtered_news_chunk["Stock_symbol"] = filtered_news_chunk[
            "Stock_symbol"
        ].astype(str)
        filtered_news_chunk["Url"] = filtered_news_chunk["Url"].astype(str)
        filtered_news_chunk["Author"] = filtered_news_chunk["Author"].astype(str)
        filtered_news_chunk["Article"] = filtered_news_chunk["Article"].astype(str)
        filtered_news_chunk["Lsa_summary"] = filtered_news_chunk["Lsa_summary"].astype(
            str
        )
        filtered_news_chunk["Luhn_summary"] = filtered_news_chunk[
            "Luhn_summary"
        ].astype(str)
        filtered_news_chunk["Textrank_summary"] = filtered_news_chunk[
            "Textrank_summary"
        ].astype(str)
        filtered_news_chunk["Lexrank_summary"] = filtered_news_chunk[
            "Lexrank_summary"
        ].astype(str)
        if len(filtered_news_chunk) == 0:
            continue
        table = pa.Table.from_pandas(filtered_news_chunk)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    # 1. Download FNSPID dataset from Huggingface
    data_dir_path = os.path.join(os.path.dirname(__file__), "../../data")
    os.makedirs(data_dir_path, exist_ok=True)
    urls = [
        "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv",
    ]

    for url in urls:
        print("Downloading:", url)
        out_path = os.path.join(data_dir_path, os.path.basename(url))

        download_with_resume(url, out_path)
        print("done:", out_path, os.path.getsize(out_path))

    # 2. Extract ticker in sp500
    sp500_ticker_pd = pd.read_csv(
        os.path.join(
            data_dir_path, "S&P 500 Historical Components & Changes(11-16-2025).csv"
        )
    )

    sp500_ticker_pd["tickers"] = (
        sp500_ticker_pd["tickers"].astype(str).apply(lambda x: np.array(x.split(",")))
    )
    ticker_sp = np.unique(np.concatenate(sp500_ticker_pd["tickers"].to_numpy()))

    financial_news_path = os.path.join(data_dir_path, "financial_news")
    os.makedirs(financial_news_path, exist_ok=True)

    # 3. Extract financial news for sp500 tickers
    extrace_data_set_by_symbols(
        symbol_list=ticker_sp.tolist(),
        news_db_path=os.path.join(data_dir_path, "nasdaq_exteral_data.csv"),
        output_path=os.path.join(financial_news_path, "sp500_financial_news.parquet"),
    )
