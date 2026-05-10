# Clark-style SV first; Primiceri-style deferred to a future adapter

The first concrete `StochasticVolatility` adapter is Clark-style: per-variable log-volatility (AR(1) or random walk on `h_i,t`) with a constant correlation Cholesky factor `R_chol`. Primiceri-style time-varying-correlation SV is deferred to a future `VolatilityProcess` adapter (planned, not scheduled).

**Why**: Empirically, Clark (2011) and follow-up work consistently show that variance-only SV-VAR forecasts as well or better than full TVP-VAR-with-SV in macro datasets — parsimony wins for forecasting. Time-varying correlations add substantial latent state (`n*(n-1)/2` random walks for the elements of `A_t`) with often-poor mixing. Whether to support TVP correlations is a research call ("Primiceri shop vs Clark shop") best made deliberately, when evidence demands it.

The `VolatilityProcess` seam is shaped to admit Primiceri later (`L_t` is the natural primitive for both Clark and Primiceri; see ADR-0001) — shipping Clark first does not foreclose the option.
