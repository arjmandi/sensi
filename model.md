	1.	Deterministic function (static map)
o_k = f(i_k). Any (possibly nonlinear) function mapping inputs to outputs.
	2.	Linear / affine model
o_k = W i_k + b (vector case) or o_k = w^\top i_k + b (scalar).
	3.	Basis expansion (poly/splines/feature maps)
o_k = \sum_{j} \alpha_j\,\phi_j(i_k), where \phi_j are polynomials, splines, Fourier, etc.
	4.	k-Nearest Neighbors (nonparametric)
\hat o(i) = \frac{1}{k}\sum_{j\in \mathcal N_k(i)} o_j.
	5.	Kernel regression (Nadaraya–Watson)
\hat o(i)=\dfrac{\sum_j K(i,i_j)\,o_j}{\sum_j K(i,i_j)}.
	6.	Logistic/softmax (classification)
\Pr(o=c\mid i)=\text{softmax}_c(W i + b).
	7.	Decision trees / ensembles (RF/GBDT)
Piecewise-constant (or -linear) partitions of the input space: o_k = \text{Tree}(i_k); ensembles average/vote.
	8.	Gaussian process (Bayesian nonparametric)
f\sim \mathcal{GP}(m,k); predict o with mean/variance from kernel k(i,i’).
	9.	Mixture of experts
o = \sum_{m=1}^M \pi_m(i)\, f_m(i), with gating \pi_m(i).
	10.	Bayesian likelihood model
Specify p(o\mid i,\theta) and prior p(\theta); infer p(\theta\mid \mathcal D), predict p(o\mid i,\mathcal D).
	11.	Noisy memoryless channel (information theory)
Model uncertainty as p(o\mid i); capacity/entropy tools apply.
	12.	Causal structural equation model (SEM / DAG)
o = g(i,u) with exogenous noise u; supports interventions do(i) and causal queries.
	13.	Linear time-invariant (LTI) / convolutional system
For sequences: o_t=\sum_{\tau=0}^{\infty} h_\tau\, i_{t-\tau}; transfer H(z) or H(\omega).
	14.	State-space model (Kalman/linear Gaussian)
x_{t+1}=A x_t+B i_t+w_t,\quad o_t=C x_t + D i_t + v_t. Inference via Kalman filter/smoother.
	15.	ARIMAX / dynamic regression
o_t = \sum_{j=1}^p \phi_j o_{t-j} + \sum_{m=0}^q \beta_m i_{t-m} + \epsilon_t (with differencing if needed).
	16.	Input–Output HMM (IO-HMM)
Latent z_t with p(z_{t+1}\!\mid z_t,i_t) and p(o_t\!\mid z_t,i_t).
	17.	Recurrent neural network (RNN/LSTM/GRU)
h_t = \text{RNN}(h_{t-1}, i_t), o_t = g(h_t).
	18.	Transformer (seq2seq / attention)
o_{1:T} = \text{Decoder}(\text{Encoder}(i_{1:T})) with self-/cross-attention.
	19.	Contextual bandit (interventions as actions)
Pick i_t (action) given context x_t; observe reward o_t\sim r(i_t,x_t); learn to maximize expected reward.
	20.	MDP / Reinforcement learning
State s_t, action i_t, reward o_t; dynamics p(s_{t+1}\!\mid s_t,i_t); optimize return \mathbb E[\sum \gamma^t o_t].


	21.	Generalized linear model (GLM)
g(\mathbb E[o\mid i]) = \beta_0 + \beta^\top i with link g (Poisson, Gamma, Tweedie, etc.).
	22.	Generalized additive model (GAM)
g(\mathbb E[o\mid i]) = \alpha + \sum_j s_j(i_j), smooth s_j via splines.
	23.	Quantile regression
Q_{o}(\tau\mid i) = \beta_0(\tau) + \beta(\tau)^\top i for chosen quantile \tau.
	24.	Support vector regression / classification
SVR: minimize \tfrac12\|w\|^2 + C\sum_k \varepsilon_k s.t. |o_k - w^\top \phi(i_k) - b|\le \epsilon+\varepsilon_k.
	25.	MARS (multivariate adaptive regression splines)
o \approx \sum_m c_m\, B_m(i) with data-driven hinge basis functions.
	26.	Isotonic (monotonic) regression
Find monotone f minimizing \sum_k (o_k - f(i_k))^2.
	27.	Penalized linear models (ridge/lasso/elastic net)
o_k \approx \beta^\top i_k with \ell_2/\ell_1 (or mixed) penalties on \beta.
	28.	Reduced-rank regression
O \approx I B with \text{rank}(B)\le r; captures low-dimensional linear mapping.
	29.	Mixture density network (MDN)
Neural net outputs \{\pi_m(i),\mu_m(i),\Sigma_m(i)\}; p(o\mid i)=\sum_m \pi_m \,\mathcal N(o;\mu_m,\Sigma_m).
	30.	Conditional normalizing flow
Invertible z=f_\theta(o;i); p(o\mid i)=p_Z(f_\theta(o;i))\left|\det \frac{\partial f_\theta}{\partial o}\right|.
	31.	Copula-based regression
Model marginals F_o, F_i and dependence via copula C: P(o\le y\mid i)=\partial C(F_o(y),F_i(i))/\partial F_i.
	32.	Cox proportional hazards (survival)
\lambda(t\mid i)=\lambda_0(t)\exp(\beta^\top i); outputs are time-to-event distributions.
	33.	Ordinal regression (proportional odds)
\Pr(o\le c\mid i)=\sigma(\theta_c - \beta^\top i) for ordered classes c.
	34.	Zero-inflated / hurdle count models
Mixture: \Pr(o=0\mid i)=\pi(i)+[1-\pi(i)]f(0\mid i), else o\sim f(\cdot\mid i) (e.g., NB/Poisson).
	35.	Conditional random field (CRF)
For sequences/structures: p(o_{1:T}\mid i_{1:T}) \propto \exp\!\big(\sum_t \theta^\top f(o_{t-1},o_t,i_{1:T},t)\big).
	36.	Energy-based model (EBM)
Define energy E_\theta(o,i); p(o\mid i)\propto e^{-E_\theta(o,i)}.
	37.	Volterra series (nonlinear convolution)
o_t = \sum_\tau h_1(\tau)i_{t-\tau} + \sum_{\tau_1,\tau_2} h_2(\tau_1,\tau_2) i_{t-\tau_1}i_{t-\tau_2} + \cdots.
	38.	Hammerstein–Wiener block models
Static nonlinearity → LTI (or LTI → nonlinearity): o_t = (H * g(i))_t (or o_t=g((H*i)_t)).
	39.	NARX (nonlinear AR with exogenous input)
o_t = F(o_{t-1:t-p},\, i_{t:t-q}) + \epsilon_t with nonlinear F (e.g., NN).
	40.	Nonlinear SDE state-space
dx_t=f(x_t,i_t)dt+G\,dW_t,\quad o_t=h(x_t)+v_t; infer with EKF/UKF/particles.
