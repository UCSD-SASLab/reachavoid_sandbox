# Sandbox Reach-Avoid Problem 
Use `uv sync` to install venv then run `uv run avoid.py` or `uv run reachavoid.py`

## avoid.py
Solves a backwards-reachable tube problem for the avoiding an obstacle $$\mathcal{F}.

Define $$\ell(x)$$ such that $$x \in \mathcal{F} \iff \ell(x) < 0$$.

The DP update equation is $$V_{k+1} = \min[\ell(x), V_k + H]$$ with $$H = \max_u \min_d \nabla V^\top f(x,u,d)$$

## reach.py
Solve a backwards-reachable tube problem for reaching (not staying in!) a target $$\mathcal{T}.

Define $$g(x)$$ such that $$x \in \mathcal{T} \iff g(x) \geq 0$$.

The DP update equation is $$V_{k+1} = \max[g(x), V_k + H]$$ with $$H = \max_u \min_d \nabla V^\top f(x,u,d)$$

## reachavoid.py
Solve a backwards reach-avoid tube problem for reaching a target while avoiding obstacles (its fine to hit an obstacle after reaching the target). 

The DP update equation is $$V_{k+1} = \max[g(x), \min(\ell(x), V_k + H)]$$ with $$H = max_u min_d \nabla V^\top f(x,u,d)$$

