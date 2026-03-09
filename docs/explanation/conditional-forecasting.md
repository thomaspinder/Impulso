# Conditional Forecasting

## The problem

A standard VAR forecast projects all variables forward simultaneously, with no external constraints. But many questions in macroeconomics and finance require **conditional** projections:

- Central banks publish forecasts conditioned on assumed interest rate paths
- Financial institutions stress-test portfolios under assumed macroeconomic scenarios
- Policy analysts ask "what if" questions about specific variable trajectories

Conditional forecasting provides the mathematical framework for these exercises.

## The idea

A VAR forecast can be decomposed into two parts:

$$y_{t+h} = \underbrace{y_{t+h}^{u}}_{\text{unconditional}} + \underbrace{\sum_{s=0}^{h} \Phi_{h-s} \, P \, \varepsilon_{s}}_{\text{shock-driven deviation}}$$

where $y^{u}_{t+h}$ is the unconditional forecast (no future shocks), $\Phi_j$ are the moving-average (MA) coefficient matrices, $P$ is the impact matrix (Cholesky factor of $\Sigma$ or a structural matrix), and $\varepsilon_s$ are the future structural shocks.

The unconditional forecast is deterministic given the posterior draw. The future shocks are unknown. Conditional forecasting amounts to **choosing the shock paths** that make the forecast satisfy the desired constraints.

## The Waggoner-Zha algorithm

Waggoner & Zha (1999) showed that this can be formulated as a linear system. Stack all future shocks into a vector $\varepsilon = [\varepsilon_0, \varepsilon_1, \ldots, \varepsilon_{H-1}]$ of length $H \times n$, where $H$ is the number of forecast steps and $n$ is the number of variables.

Each constraint (e.g., "variable $i$ equals value $v$ at period $h$") translates into a linear equation:

$$R \, \varepsilon = c$$

where $R$ is constructed from the MA coefficients and the impact matrix, and $c$ contains the differences between target values and unconditional forecasts.

The minimum-norm solution $\varepsilon^* = R^\top (R R^\top)^{-1} c$ gives the **smallest set of shocks** (in a least-squares sense) that satisfies all constraints. This is computed via `numpy.linalg.lstsq`.

The conditional forecast is then:

$$y_{t+h}^{c} = y_{t+h}^{u} + \sum_{s=0}^{h} \Phi_{h-s} \, P \, \varepsilon^*_s$$

## Structural extensions

When the model is identified (you have a structural impact matrix $P$ rather than just the Cholesky factor of $\Sigma$), you can also condition on **structural shock paths**. For example, "assume the supply shock is zero for the next 4 periods" translates into direct constraints on elements of $\varepsilon$.

Observable constraints and shock constraints can be combined in a single system. Observable constraints use the full MA representation (involving $\Phi$ and $P$), while shock constraints are simpler — they directly pin individual elements of $\varepsilon$.

## Bayesian uncertainty

The algorithm is applied independently to each posterior draw of $(B, \Sigma)$ or $(B, P)$. This means:

- The unconditional forecast differs across draws (parameter uncertainty)
- The MA coefficients differ across draws
- The shock paths satisfying the constraints differ across draws

The result is a full posterior distribution of conditional forecasts, from which you can compute medians, HDIs, and other summaries. The constrained variables will hit their targets exactly in every draw, but the unconstrained variables will show genuine posterior uncertainty about the conditional projection.
