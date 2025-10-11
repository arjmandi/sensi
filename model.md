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



