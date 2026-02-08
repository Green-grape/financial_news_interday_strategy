from datetime import datetime

import numpy as np
import pandas as pd


def backtest_sentiment_ls(
    df: pd.DataFrame,
    long_q: float = 0.9,
    short_q: float = 0.1,
    gross_exposure: float = 1.0,  # 1.0이면 long 합 +1, short 합 -1 (순노출 0)
    weight_mode: str = "equal",  # "equal" or "count"
    cost_bps_one_way: float = 5.0,  # 편도 비용(bps). 왕복이면 10bps에 해당
    min_names_per_side: int = 5,  # 너무 적은 날 제외
    target_col: str = "sentiment_score",
    weight_cap: float = None,  # 종목별 최대 비중 제한 (예: 0.1이면 10%). None이면 제한 없음
) -> dict:
    """
    df columns required:
      entry_date, Stock_symbol, sentiment_score, sentiment_count, news_return
    Assumption:
      news_return is the realized return for the holding period implied by signal day (after market -> next day return)
    """

    required = {
        "entry_date",
        "Stock_symbol",
        target_col,
        "sentiment_count",
        "news_return",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.copy()
    d["entry_date"] = pd.to_datetime(d["entry_date"]).dt.date
    d = d.dropna(subset=["entry_date", "Stock_symbol", target_col, "news_return"])
    d["Stock_symbol"] = d["Stock_symbol"].astype(str)

    # 일자별로 long/short 컷 계산
    def pick_sides(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < 2 * min_names_per_side:
            return g.assign(side=0).iloc[0:0]  # empty
        hi = g[target_col].quantile(long_q)
        if short_q > 0.0:
            lo = g[target_col].quantile(short_q)
            print(f"lo cutoff: {lo}")
        else:
            lo = -np.inf  # No short side
        gg = g.copy()
        gg["side"] = 0
        gg.loc[gg[target_col] >= hi, "side"] = 1
        gg.loc[gg[target_col] <= lo, "side"] = -1
        gg = gg[gg["side"] != 0]

        # side별 종목수 체크
        if (gg["side"] == 1).sum() < min_names_per_side:
            return gg.iloc[0:0]
        if lo != -np.inf and (gg["side"] == -1).sum() < min_names_per_side:
            return gg.iloc[0:0]
        return gg

    picks = d.groupby("entry_date", group_keys=False).apply(pick_sides)
    if picks.empty:
        raise ValueError(
            "No trading days after filtering. Check quantiles / min_names_per_side / data."
        )

    # 가중치 부여: long은 +gross_exposure, short은 -gross_exposure
    def assign_weights(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        long_mask = g["side"] == 1
        short_mask = g["side"] == -1

        if weight_mode == "equal":
            if long_mask.any():
                g.loc[long_mask, "w_raw"] = 1.0
            if short_mask.any():
                g.loc[short_mask, "w_raw"] = 1.0
        elif weight_mode == "count":
            g.loc[long_mask, "w_raw"] = (
                g.loc[long_mask, "sentiment_count"].clip(lower=1).astype(float)
            )
            g.loc[short_mask, "w_raw"] = (
                g.loc[short_mask, "sentiment_count"].clip(lower=1).astype(float)
            )
        elif weight_mode == "score":
            g.loc[long_mask, "w_raw"] = (
                g.loc[long_mask, target_col].clip(lower=0.0).astype(float)
            )
            g.loc[short_mask, "w_raw"] = (
                (-g.loc[short_mask, target_col]).clip(lower=0.0).astype(float)
            )
        else:
            raise ValueError("weight_mode must be 'equal' or 'count' or 'score'")

        # Normalize to gross_exposure
        if long_mask.any():
            s = g.loc[long_mask, "w_raw"].sum()
            g.loc[long_mask, "w"] = (g.loc[long_mask, "w_raw"] / s) * gross_exposure
            if weight_cap is not None:
                g.loc[long_mask, "w"] = g.loc[long_mask, "w"].clip(upper=weight_cap)

                s2 = g.loc[long_mask, "w"].sum()
                g.loc[long_mask, "w"] = (g.loc[long_mask, "w"] / s2) * gross_exposure
        if short_mask.any():
            s = g.loc[short_mask, "w_raw"].sum()
            g.loc[short_mask, "w"] = -(g.loc[short_mask, "w_raw"] / s) * gross_exposure
            if weight_cap is not None:
                g.loc[short_mask, "w"] = g.loc[short_mask, "w"].clip(lower=-weight_cap)

                s2 = -g.loc[short_mask, "w"].sum()
                g.loc[short_mask, "w"] = -(g.loc[short_mask, "w"] / s2) * gross_exposure
        return g.drop(columns=["w_raw"])

    weights = picks.groupby("entry_date", group_keys=False).apply(assign_weights)

    # (entry_date, symbol) -> weight matrix
    W = (
        weights.pivot_table(
            index="entry_date", columns="Stock_symbol", values="w", aggfunc="sum"
        )
        .fillna(0.0)
        .sort_index()
    )

    # (entry_date, symbol) -> realized return
    R = (
        d.pivot_table(
            index="entry_date",
            columns="Stock_symbol",
            values="news_return",
            aggfunc="mean",
        )
        .reindex(W.index)
        .fillna(0.0)
    )

    # 일별 총수익(거래비용 전): sum_i w_{t,i} * r_{t,i}
    gross_ret = (W * R).sum(axis=1)

    # turnover 기반 거래비용
    # 거래비용은 보통 "달러 거래량"에 비례 → sum |Δw|
    dW = W.diff().abs()
    dW.iloc[0] = W.iloc[0].abs()  # 첫날은 진입비용 발생한다고 가정

    # 비용 단가: bps -> decimal
    c = cost_bps_one_way / 10000.0

    turnover = 2.0 * np.ones_like(gross_ret)  # 무조건 포지션을 청산하는 경우

    # if include_short_cost:
    #     turnover = dW.sum(axis=1)
    # else:
    #     # long leg에 대해서만 비용 적용(비현실적일 수 있으나 옵션)
    #     turnover = dW.where(W.shift(1).fillna(0.0) >= 0, 0.0).sum(axis=1)

    cost = turnover * c
    net_ret = gross_ret - cost

    cum_net_ret = pd.Series((1 + net_ret).cumprod())

    if cum_net_ret.empty:
        maximum_drawdown = 0.0
    else:
        temp = pd.DataFrame(
            {
                "hwm": np.maximum.accumulate(cum_net_ret.values),
                "pnl": cum_net_ret.values,
            },
            index=cum_net_ret.index,
        )

        max_min_df = temp.groupby("hwm")["pnl"].min().reset_index()
        max_min_df.columns = ["hwm", "min_pnl"]
        max_min_df.index = temp["hwm"].drop_duplicates(keep="first").index
        max_min_df = max_min_df[max_min_df["hwm"] > max_min_df["min_pnl"]]
        dd = (max_min_df["hwm"] - max_min_df["min_pnl"]) / max_min_df["hwm"]
        maximum_drawdown = dd.max() if not dd.empty else 0.0

    # 요약 통계
    def ann_sharpe(x: pd.Series, periods=252):
        mu = x.mean()
        sd = x.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return np.nan
        return (mu / sd) * np.sqrt(periods)

    equity = (1.0 + net_ret).cumprod()
    stats = {
        "days": int(net_ret.shape[0]),
        "avg_daily_gross": float(gross_ret.mean()),
        "avg_daily_cost": float(cost.mean()),
        "avg_daily_net": float(net_ret.mean()),
        "vol_daily_net": float(net_ret.std(ddof=1)),
        "sharpe_net": float(ann_sharpe(net_ret)),
        "cum_return_net": float(equity.iloc[-1] - 1.0),
        "avg_turnover": float(turnover.mean()),
        "maximum_drawdown": float(maximum_drawdown),
    }

    out = pd.DataFrame(
        {
            "gross_ret": gross_ret,
            "turnover": turnover,
            "cost": cost,
            "net_ret": net_ret,
            "equity_net": equity,
        }
    )

    return {"daily": out, "weights": W, "stats": stats}
