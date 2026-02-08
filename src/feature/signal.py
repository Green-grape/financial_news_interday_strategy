import os
from collections import deque

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import pipeline


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS device")
        return "mps"
    else:
        return "cpu"


def parse_emb_str(s):
    s = s.strip()
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    return np.fromstring(s, sep=" ", dtype=np.float32)


def normalize_vec(v):
    n = np.linalg.norm(v + 1e-10)
    return v / n


def dedup_by_symbol(dataset_path, output_path, threshold=0.8, window_days=4):
    dataset = ds.dataset(dataset_path, format="parquet")

    # 종목 목록만 먼저 수집(컬럼이 dictionary/low-cardinality면 더 빠름)
    syms = (
        dataset.to_table(columns=["Stock_symbol"])
        .column("Stock_symbol")
        .unique()
        .to_pylist()
    )

    writer = None
    for sym in tqdm(syms, desc="Processing symbols"):
        table = dataset.to_table(
            filter=pc.equal(pc.field("Stock_symbol"), sym),
            columns=[
                "Date",
                "Stock_symbol",
                "article_title_embedding",
                "article_title_pos_score",
                "article_title_neg_score",
                "article_title_neu_score",
            ],
        )
        df = table.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date"])

        # 여기서는 기존 함수 그대로(종목 하나니까 groupby 불필요)
        out = []
        window = deque()
        for idx, row in df.iterrows():
            cur_date = row["Date"]
            cur_emb = normalize_vec(parse_emb_str(row["article_title_embedding"]))
            while window and (cur_date - window[0][0]).days > window_days:
                window.popleft()

            if any(np.dot(cur_emb, e) >= threshold for _, e in window):
                continue
            out.append(row)
            window.append((cur_date, cur_emb))

        if not out:
            continue
        out_df = pd.DataFrame(out)
        table_out = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                output_path, table_out.schema, compression="snappy"
            )
        writer.write_table(table_out)

    if writer:
        writer.close()


if __name__ == "__main__":
    # 1. Set Classification Model
    task = "text-classification"
    model_id = "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"

    classifier = pipeline(task, model_id, device=get_device(), top_k=None)

    data_dir = os.path.join(os.path.dirname(__file__), "../../data")
    processed_dir = os.path.join(data_dir, "financial_news/processed")
    news_data_path = os.path.join(processed_dir, "sp500_financial_news.parquet")

    pq_file = pq.ParquetFile(news_data_path)

    new_score_data_path = os.path.join(
        processed_dir, "sp500_financial_news_sentiment_title.parquet"
    )

    writer = None
    # 2. Inference Sentiment Score and Save
    for chunk_idx, news_chunk in tqdm(
        enumerate(pq_file.iter_batches(batch_size=100000))
    ):
        batch_size = 16  # Adjust based on your GPU memory
        news_chunk = news_chunk.dropna(subset=["Article_title"])
        texts = news_chunk["Article_title"].tolist()
        if len(texts) == 0:
            continue
        results = classifier(
            texts, return_all_scores=True, batch_size=batch_size, show_progress_bar=True
        )
        pos_scores = []
        neg_scores = []
        neu_scores = []
        for res in results:
            for r in res:
                if r["label"] == "positive":
                    pos_scores.append(r["score"])
                elif r["label"] == "negative":
                    neg_scores.append(r["score"])
                elif r["label"] == "neutral":
                    neu_scores.append(r["score"])

        cur_df = news_chunk[["Date", "Article_title", "Stock_symbol"]].assign(
            article_pos_score=pos_scores,
            article_neg_score=neg_scores,
            article_neu_score=neu_scores,
        )
        table = pa.Table.from_pandas(cur_df)
        if writer is None:
            writer = pq.ParquetWriter(new_score_data_path, table.schema)
        writer.write_table(table)
        print(f"Processed chunk {chunk_idx}, wrote {len(cur_df)} rows.")
    if writer is not None:
        writer.close()

    # 3. Apply Sentiment embedding
    import numpy as np
    from sentence_transformers import SentenceTransformer

    new_score_data_path_with_embeddings = os.path.join(
        processed_dir, "sp500_financial_news_sentiment_title_with_embeddings.parquet"
    )
    model = SentenceTransformer("FinLang/finance-embeddings-investopedia")
    pq_file = pq.ParquetFile(new_score_data_path)
    writer = None
    for chunk_idx, news_chunk in tqdm(
        enumerate(pq_file.iter_batches(batch_size=100000))
    ):
        embeddings = model.encode(
            news_chunk["Article_title"].tolist(), batch_size=16, show_progress_bar=True
        )
        news_chunk["article_title_embedding"] = list(embeddings)
        table = pa.Table.from_pandas(news_chunk)
        if writer is None:
            writer = pq.ParquetWriter(new_score_data_path_with_embeddings, table.schema)
        writer.write_table(table)
        print(f"Processed chunk {chunk_idx}, wrote {len(news_chunk)} rows.")
    if writer is not None:
        writer.close()

    # 4. Delete Redundant News by Embedding Similarity
    cos_thresh = 0.8  # cosine similarity threshold(By 'Sentiment trading with large language models' paper, 0.8 is used)
    win_days = 4  # time window in days (By 'Predicting Returns with Text Data' paper, 4 days is Speed of news assimilation)
    dedup_by_symbol(
        dataset_path=os.path.join(
            processed_dir,
            "sp500_financial_news_sentiment_title_with_embeddings.parquet",
        ),
        output_path=os.path.join(
            processed_dir,
            f"sp500_financial_news_sentiment_deduped_{cos_thresh}_{win_days}.parquet",
        ),
        threshold=cos_thresh,
        window_days=win_days,
    )
