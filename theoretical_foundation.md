# Sentiment-Driven Stock Trend Prediction: Theoretical & Mathematical Foundation

> **Authors**: Çulban & Bayraktar (2026)  
> **Document Type**: Graduation Project - Theoretical Foundation  
> **Version**: 1.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Financial Market Theory](#2-financial-market-theory)
3. [Statistical Framework for Time Series Analysis](#3-statistical-framework-for-time-series-analysis)
4. [Sentiment Quantification Theory](#4-sentiment-quantification-theory)
5. [Natural Language Processing Mathematics](#5-natural-language-processing-mathematics)
6. [Deep Learning Mathematical Foundations](#6-deep-learning-mathematical-foundations)
7. [Multimodal Fusion Theory](#7-multimodal-fusion-theory)
8. [Noise and Reliability Quantification](#8-noise-and-reliability-quantification)
9. [Cross-Market Comparative Framework](#9-cross-market-comparative-framework)
10. [Evaluation Metrics and Statistical Validation](#10-evaluation-metrics-and-statistical-validation)

---

## 1. Executive Summary

This document establishes the rigorous theoretical and mathematical foundations for a sentiment-driven stock trend prediction system. The framework integrates three major domains:

1. **Quantitative Finance**: Market microstructure, noise trader theory, and behavioral finance
2. **Natural Language Processing**: Transformer architectures, attention mechanisms, and sentiment embeddings
3. **Deep Learning**: Recurrent neural networks, multimodal fusion, and uncertainty quantification

The key innovation lies in treating sentiment not merely as an input feature but as a signal with quantifiable reliability, enabling the model to dynamically adjust its trust in social media signals based on market conditions.

---

## 2. Financial Market Theory

### 2.1 Efficient Market Hypothesis (EMH)

The classical framework assumes markets are informationally efficient. Under EMH, asset prices follow a random walk:

$$P_t = P_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

where $\epsilon_t$ is white noise representing unpredictable information arrival.

**Three Forms of Efficiency**:

| Form | Information Set | Implication |
|------|-----------------|-------------|
| Weak | Historical prices | Technical analysis fails |
| Semi-strong | All public information | Fundamental analysis fails |
| Strong | All information (including private) | No strategy works |

### 2.2 Adaptive Markets Hypothesis (AMH)

Lo (2004) proposed that market efficiency is not constant but evolves based on competitive dynamics:

$$\eta_t = f(\text{Competition}_t, \text{Arbitrage Capacity}_t, \text{Information Speed}_t)$$

where $\eta_t \in [0, 1]$ represents the degree of market efficiency at time $t$.

**Key Insight**: During high volatility or information cascades (e.g., viral tweets), $\eta_t$ decreases, creating exploitable inefficiencies that sentiment analysis can capture.

### 2.3 Noise Trader Theory (De Long et al., 1990)

The foundational model for our approach. Define two types of investors:

- **Informed Traders (I)**: Trade on fundamental value $V$
- **Noise Traders (N)**: Trade on misperception $\rho_t$

The price of a risky asset becomes:

$$P_t = V + \frac{\mu_N}{1 + r}\rho_t - \frac{\gamma \sigma_\rho^2 \mu_N^2}{(1+r)^2}$$

where:
- $\mu_N$ = proportion of noise traders
- $\rho_t$ = average misperception of noise traders (sentiment)
- $r$ = risk-free rate
- $\gamma$ = risk aversion coefficient
- $\sigma_\rho^2$ = variance of noise trader sentiment

**Theorem 2.1 (Noise Trader Survival)**: Noise traders can survive and even prosper if:

$$E[\rho_t] > \gamma \sigma_\rho^2$$

*Interpretation*: If noise traders are systematically optimistic and the market doesn't immediately punish them, sentiment-driven mispricing persists.

### 2.4 Limits to Arbitrage

Rational arbitrageurs face constraints that prevent immediate price correction:

**Fundamental Risk**: Even if $P > V$, the fundamental value $V$ might fall before price corrects.

**Noise Trader Risk**: Even if correct, arbitrageurs face the risk that:

$$P_{t+1} = P_t + \Delta\rho_{t+1}$$

where $\Delta\rho_{t+1}$ could move prices further from fundamentals.

**Capital Constraints**: Arbitrage capital $K$ is limited:

$$\max_{\alpha} E[U(W)] \quad \text{s.t.} \quad \alpha_t \cdot |P_t - V| \leq K$$

### 2.5 Information Cascade Model

When investors observe others' actions rather than their own signals:

$$P(\text{Buy}|\text{Signal}=s, H_t) = \frac{P(H_t|\text{Buy}) \cdot P_0(\text{Buy}|s)}{P(H_t)}$$

where $H_t$ is the history of observed trades.

**Cascade Formation**: A cascade occurs when:

$$\left|\sum_{i=1}^{t} \text{sign}(\text{action}_i)\right| > \tau$$

In social media, this manifests as viral posts creating herding behavior independent of fundamental information.

---

## 3. Statistical Framework for Time Series Analysis

### 3.1 Stationarity and Unit Root Tests

Financial time series must be tested for stationarity before modeling.

**Augmented Dickey-Fuller (ADF) Test**:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$$

- $H_0$: $\gamma = 0$ (unit root, non-stationary)
- $H_1$: $\gamma < 0$ (stationary)

**Phillips-Perron Test**: Robust to heteroskedasticity and serial correlation.

**KPSS Test**: $H_0$ is stationarity (complementary to ADF).

### 3.2 Cointegration Analysis

For sentiment $S_t$ and price $P_t$ to have a long-run relationship:

**Engle-Granger Two-Step**:

1. Estimate: $P_t = \alpha + \beta S_t + u_t$
2. Test residuals $\hat{u}_t$ for stationarity

If $\hat{u}_t$ is I(0), then $P_t$ and $S_t$ are cointegrated.

**Johansen Test** (for multivariate cointegration):

$$\Delta X_t = \Pi X_{t-1} + \sum_{i=1}^{k-1} \Gamma_i \Delta X_{t-i} + \epsilon_t$$

where $\Pi = \alpha\beta'$, and $\text{rank}(\Pi) = r$ gives the number of cointegrating relationships.

### 3.3 Granger Causality

To test if sentiment "Granger-causes" price movements:

$$P_t = \sum_{i=1}^{p} \alpha_i P_{t-i} + \sum_{j=1}^{q} \beta_j S_{t-j} + \epsilon_t$$

- $H_0$: $\beta_1 = \beta_2 = ... = \beta_q = 0$ (no Granger causality)
- Test statistic: $F = \frac{(SSR_R - SSR_U)/q}{SSR_U/(T-p-q-1)}$

**Bidirectional Causality**: Also test $S_t$ on lagged $P_t$ to detect feedback loops.

### 3.4 Volatility Modeling (GARCH Family)

Stock returns exhibit volatility clustering which must be modeled.

**GARCH(p,q)**:

$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1)$$

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2$$

**EGARCH** (asymmetric effects):

$$\log(\sigma_t^2) = \omega + \sum_{i=1}^{q} \left[\alpha_i |z_{t-i}| + \gamma_i z_{t-i}\right] + \sum_{j=1}^{p} \beta_j \log(\sigma_{t-j}^2)$$

where $\gamma_i < 0$ captures the leverage effect (negative news increases volatility more than positive news).

**Sentiment-Extended GARCH**:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 + \lambda \cdot \text{SentimentVolatility}_{t-1}$$

This tests if sentiment disagreement predicts return volatility.

### 3.5 Vector Autoregression (VAR)

For simultaneous modeling of sentiment and price:

$$\begin{bmatrix} P_t \\ S_t \\ V_t \end{bmatrix} = \sum_{i=1}^{p} A_i \begin{bmatrix} P_{t-i} \\ S_{t-i} \\ V_{t-i} \end{bmatrix} + \begin{bmatrix} \epsilon_{1t} \\ \epsilon_{2t} \\ \epsilon_{3t} \end{bmatrix}$$

where $V_t$ = volume. This captures cross-dependencies and feedback effects.

**Impulse Response Functions (IRF)**: Trace the effect of a sentiment shock:

$$\frac{\partial P_{t+h}}{\partial \epsilon_{S,t}}$$

---

## 4. Sentiment Quantification Theory

### 4.1 Formal Definition of Sentiment

Let $\mathcal{D} = \{d_1, d_2, ..., d_n\}$ be a corpus of documents (tweets, news).

**Definition 4.1 (Sentiment Function)**: A sentiment function $s: \mathcal{D} \to \mathbb{R}^k$ maps documents to a $k$-dimensional sentiment space, where:

- $k = 1$: Univariate polarity (positive/negative)
- $k = 3$: Polarity, intensity, subjectivity
- $k = d_{embed}$: Full embedding representation

### 4.2 Lexicon-Based Sentiment (VADER)

VADER assigns valence scores using a lexicon $L = \{(w_i, v_i)\}$:

$$\text{Compound} = \frac{\sum_{i \in d} v_i}{\sqrt{\left(\sum_{i \in d} v_i\right)^2 + \alpha}}$$

where $\alpha = 15$ is a normalization constant.

**Limitations**:
- Cannot handle negation well ("not good")
- Misses financial idioms ("dead cat bounce" = slightly positive in VADER)
- No contextual understanding

### 4.3 Transformer-Based Sentiment (FinBERT)

**Architecture**: BERT with domain-specific pre-training on financial corpora.

**Sentiment Probability**:

$$P(y|d) = \text{softmax}(W \cdot h_{[CLS]} + b)$$

where $h_{[CLS]} \in \mathbb{R}^{768}$ is the [CLS] token representation.

**Advantage**: Contextual understanding:
- "Apple crushed expectations" → Positive (earnings context)
- "Apple crushed employees" → Negative (labor context)

### 4.4 Aggregation Strategies

For multiple documents on day $t$:

**Simple Average**:
$$S_t = \frac{1}{|D_t|} \sum_{d \in D_t} s(d)$$

**Volume-Weighted**:
$$S_t^{VW} = \frac{\sum_{d \in D_t} w_d \cdot s(d)}{\sum_{d \in D_t} w_d}$$

where $w_d$ = engagement metrics (retweets, replies).

**Time-Decayed**:
$$S_t^{TD} = \sum_{d \in D_t} s(d) \cdot e^{-\lambda(t_{close} - t_d)}$$

where $\lambda$ controls decay rate and $(t_{close} - t_d)$ is time until market close.

**Reliability-Weighted** (Novel):
$$S_t^{RW} = \frac{\sum_{d \in D_t} R_d \cdot s(d)}{\sum_{d \in D_t} R_d}$$

where $R_d$ is a reliability score based on source credibility, historical accuracy, and herding indicators.

### 4.5 Sentiment Dispersion

Beyond average sentiment, dispersion captures disagreement:

$$\text{Dispersion}_t = \sqrt{\frac{1}{|D_t|} \sum_{d \in D_t} (s(d) - S_t)^2}$$

**Hypothesis**: High dispersion indicates uncertainty and often precedes volatility.

---

## 5. Natural Language Processing Mathematics

### 5.1 Word Embeddings

**Word2Vec Skip-gram Objective**:

$$\mathcal{L} = \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j}|w_t)$$

$$P(w_O|w_I) = \frac{\exp(v'_{w_O} \cdot v_{w_I})}{\sum_{w \in V} \exp(v'_w \cdot v_{w_I})}$$

**Financial Embeddings**: Pre-train on financial corpora so that:
$$\cos(v_{\text{bull}}, v_{\text{up}}) > \cos(v_{\text{bull}}, v_{\text{down}})$$

### 5.2 Transformer Architecture

The core innovation enabling modern NLP.

**Self-Attention Mechanism**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q = XW^Q$ (Queries)
- $K = XW^K$ (Keys)
- $V = XW^V$ (Values)
- $d_k$ = key dimension (scaling prevents gradient vanishing)

**Multi-Head Attention**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Computational Complexity**: $O(n^2 \cdot d)$ where $n$ = sequence length.

### 5.3 BERT Architecture

**Masked Language Model (MLM)**:

$$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})$$

where $\mathcal{M}$ is the set of masked positions.

**Next Sentence Prediction (NSP)**:

$$\mathcal{L}_{NSP} = -\log P(\text{IsNext} | h_{[CLS]})$$

**FinBERT Fine-tuning**: Starting from BERT weights, minimize:

$$\mathcal{L}_{sentiment} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$

### 5.4 Turkish NLP Considerations

Turkish is an agglutinative language with complex morphology.

**Example**: "düşmeyebilirlerdi" (they might not have been able to fall)

Morphological components: düş + me + y + ebil + ir + ler + di

**BERTurk**: Uses WordPiece tokenization adapted for Turkish morphology:
- Token: `düş##me##yebilirlerdi` (subword segmentation)
- Preserves semantic meaning across morphemes

**Domain Adaptation**:

$$\mathcal{L}_{adapt} = \mathcal{L}_{MLM}^{financial} + \lambda \mathcal{L}_{sentiment}^{labeled}$$

### 5.5 Sentence Embeddings for Aggregation

Instead of averaging word embeddings, use sentence-level:

**Sentence-BERT**:

$$\text{sim}(s_1, s_2) = \cos(BERT(s_1), BERT(s_2))$$

This enables clustering similar tweets and identifying information redundancy.

---

## 6. Deep Learning Mathematical Foundations

### 6.1 Recurrent Neural Networks (RNN)

**Standard RNN**:

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$

$$y_t = W_{hy}h_t + b_y$$

**Problem**: Vanishing/exploding gradients:

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=1}^{T-1} \frac{\partial h_{t+1}}{\partial h_t}$$

If $\left\|\frac{\partial h_{t+1}}{\partial h_t}\right\| < 1$ repeatedly, gradients vanish.

### 6.2 Long Short-Term Memory (LSTM)

LSTM solves the vanishing gradient problem with gating mechanisms.

**Cell State Update**:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Gates**:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)}$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)}$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)}$$

**Candidate Memory**:

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**Hidden State**:

$$h_t = o_t \odot \tanh(c_t)$$

**Gradient Flow**: The cell state $c_t$ provides a "highway" for gradient flow:

$$\frac{\partial c_T}{\partial c_1} = \prod_{t=1}^{T-1} f_{t+1}$$

If $f_t \approx 1$ (forget gate open), gradients flow unimpeded.

### 6.3 Gated Recurrent Unit (GRU)

Simplified variant with fewer parameters.

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(Update gate)}$$

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(Reset gate)}$$

$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Trade-off**: Fewer parameters (faster training) vs. less expressive power.

### 6.4 Bidirectional RNNs

Capture both past and future context:

$$\overrightarrow{h}_t = \text{LSTM}_{\rightarrow}(x_t, \overrightarrow{h}_{t-1})$$

$$\overleftarrow{h}_t = \text{LSTM}_{\leftarrow}(x_t, \overleftarrow{h}_{t+1})$$

$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

**Application**: Processing sentiment sequences where future context matters.

### 6.5 Attention Mechanisms in Sequence Models

**Bahdanau Attention**:

$$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'=1}^{S} \exp(e_{t,s'})}$$

$$e_{t,s} = v^T \tanh(W_h h_s + W_c c_t + b)$$

$$\text{context}_t = \sum_{s=1}^{S} \alpha_{t,s} h_s$$

**Interpretation**: The model learns which time steps are most relevant for current prediction.

### 6.6 Regularization Techniques

**Dropout**: During training, randomly zero activations:

$$\tilde{h}_i = \frac{1}{1-p} \cdot h_i \cdot m_i, \quad m_i \sim \text{Bernoulli}(1-p)$$

**Recurrent Dropout**: Apply same mask across time steps (Gal & Ghahramani, 2016).

**L2 Regularization**:

$$\mathcal{L}_{reg} = \mathcal{L}_{task} + \lambda \|W\|_2^2$$

**Early Stopping**: Monitor validation loss and stop when:

$$\mathcal{L}_{val}^{(t)} > \mathcal{L}_{val}^{(t-\text{patience})}$$

### 6.7 Learning Rate Scheduling

**Cosine Annealing**:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)$$

**Warm-up**: Start with low learning rate and increase:

$$\eta_t = \eta_{target} \cdot \frac{t}{T_{warmup}}, \quad t < T_{warmup}$$

---

## 7. Multimodal Fusion Theory

### 7.1 Fusion Strategies

**Early Fusion**:

$$x_{fused} = [x_{price}; x_{sentiment}]$$

Concatenate before processing. Simple but may dilute signal.

**Late Fusion**:

$$h_{fused} = f([h_{price}; h_{sentiment}])$$

Process modalities separately, then combine representations.

**Attention-Based Fusion** (Recommended):

$$\alpha_{m} = \text{softmax}(W_a [h_{price}; h_{sentiment}; h_{context}])$$

$$h_{fused} = \alpha_{price} \cdot h_{price} + \alpha_{sentiment} \cdot h_{sentiment}$$

The weights $\alpha$ are learned and vary per time step.

### 7.2 Cross-Modal Attention

Allow price features to attend to sentiment and vice versa:

$$h'_{price} = \text{MultiHead}(h_{price}, h_{sentiment}, h_{sentiment})$$

$$h'_{sentiment} = \text{MultiHead}(h_{sentiment}, h_{price}, h_{price})$$

This captures interactions like "how does today's sentiment relate to yesterday's price action?"

### 7.3 Temporal Alignment

Price and sentiment have different temporal resolutions:
- Price: Daily OHLCV (synchronous)
- Sentiment: Continuous stream (asynchronous)

**Alignment Function**:

$$S_{aligned,t} = g\left(\{s(d) : d \in D_{t-1}^{close \to open} \cup D_t^{open \to close}\}\right)$$

where $g$ is the aggregation function (Section 4.4).

**Temporal Convolution**: Apply 1D convolution across time:

$$S_{smoothed,t} = \sum_{k=-K}^{K} w_k \cdot S_{t+k}$$

### 7.4 Proposed Architecture: Dual-Branch Network

```
Price Branch                 Sentiment Branch
    │                              │
[OHLCV + Technical]         [FinBERT Embeddings]
    │                              │
  LSTM(128)                    LSTM(128)
    │                              │
  Dropout(0.3)                 Dropout(0.3)
    │                              │
    └──────────┬───────────────────┘
               │
        Cross-Modal Attention
               │
          Dense(256)
               │
          Dropout(0.3)
               │
    ┌──────────┴──────────┐
    │                     │
 Dense(3)              Dense(1)
 (Trend Class)      (Confidence)
    │                     │
 Softmax              Sigmoid
```

**Output**:
- Trend: $\hat{y} \in \{Up, Down, Neutral\}$
- Confidence: $c \in [0, 1]$

---

## 8. Noise and Reliability Quantification

### 8.1 Herding Detection via CSAD

**Cross-Sectional Absolute Deviation**:

$$CSAD_t = \frac{1}{N} \sum_{i=1}^{N} |R_{i,t} - R_{m,t}|$$

**Herding Regression**:

$$CSAD_t = \alpha + \gamma_1 |R_{m,t}| + \gamma_2 R_{m,t}^2 + \epsilon_t$$

**Interpretation**:
- Under rational expectations: $\gamma_2 = 0$ (linear relationship)
- Under herding: $\gamma_2 < 0$ (dispersion decreases in extreme markets)

**Statistical Test**:

$$t = \frac{\hat{\gamma}_2}{\text{SE}(\hat{\gamma}_2)} \sim t_{T-3}$$

### 8.2 Sentiment-Price Divergence Index

$$DI_t = \text{Zscore}(S_t) - \text{Zscore}(R_t)$$

where Z-score normalization is:

$$\text{Zscore}(x) = \frac{x - \mu_x}{\sigma_x}$$

**Regime Classification**:

| DI Range | Interpretation | Action |
|----------|---------------|--------|
| $DI \in [-0.5, 0.5]$ | Alignment | Trust sentiment |
| $DI > 1.0$ | Bull trap (bullish sentiment, weak price) | Contrarian signal |
| $DI < -1.0$ | Wall of Worry (bearish sentiment, rising price) | Buy signal |

### 8.3 Smart Money Flow Index (SMFI)

**Daily Calculation**:

$$\Delta SMFI_t = (P_{close,t} - P_{open,t}) - (P_{high,t} - P_{low,t})$$

$$SMFI_t = SMFI_{t-1} + \Delta SMFI_t$$

**Interpretation**:
- Rising SMFI: Smart money accumulating (buying at close, selling at open)
- Falling SMFI: Smart money distributing

**Integration with Sentiment**:

$$Reliability_t = \begin{cases} 
\text{High} & \text{if } \text{sign}(S_t) = \text{sign}(\Delta SMFI_t) \\
\text{Low} & \text{otherwise}
\end{cases}$$

### 8.4 Bayesian Reliability Estimation

Model reliability as a latent variable:

$$P(R = reliable | Data) = \frac{P(Data | R = reliable) \cdot P(R = reliable)}{P(Data)}$$

**Prior**: $P(R = reliable) = 0.5$ (uninformative)

**Likelihood**: Based on historical sentiment-return correlation:

$$P(Data | R) = \prod_{t} P(R_{t+1} | S_t, R)$$

**Posterior Update**:

$$P(R_{t+1} | D_{1:t}) \propto P(observed\_return | S_t, R) \cdot P(R_t | D_{1:t-1})$$

### 8.5 Information Entropy as Noise Measure

**Sentiment Distribution Entropy**:

$$H(S_t) = -\sum_{c \in \{neg, neu, pos\}} P(c|D_t) \log P(c|D_t)$$

where $P(c|D_t)$ is the proportion of documents with class $c$ on day $t$.

- $H = 0$: Perfect agreement (could be herding)
- $H = \log(3) \approx 1.1$: Maximum disagreement

**Conditional Entropy** (given price):

$$H(S_t | R_t) = H(S_t, R_t) - H(R_t)$$

Low conditional entropy means sentiment is predictable from price (lagging indicator, less informative).

---

## 9. Cross-Market Comparative Framework

### 9.1 Market Efficiency Comparison

**Variance Ratio Test** (Lo & MacKinlay):

$$VR(q) = \frac{Var(R_t^{(q)})}{q \cdot Var(R_t^{(1)})}$$

where $R_t^{(q)}$ is the $q$-period return.

Under random walk: $VR(q) = 1$

- $VR(q) > 1$: Positive autocorrelation (momentum)
- $VR(q) < 1$: Mean reversion

**Hypothesis**: BIST will show higher deviation from $VR = 1$, indicating lower efficiency.

### 9.2 Sentiment Elasticity

**Definition**: The price response to a unit change in sentiment:

$$\epsilon_S = \frac{\partial R_{t+1}}{\partial S_t}$$

Estimated via regression:

$$R_{t+1} = \alpha + \beta S_t + \gamma X_t + \epsilon_t$$

where $X_t$ are control variables (volume, volatility).

**Comparison**:

$$\Delta \epsilon = \epsilon_S^{BIST} - \epsilon_S^{NASDAQ}$$

Test: $H_0: \Delta \epsilon = 0$ vs $H_1: \Delta \epsilon \neq 0$

### 9.3 Lead-Lag Relationships

**Cross-Correlation Function**:

$$\rho_{SR}(k) = \text{Corr}(S_t, R_{t+k})$$

- $k > 0$: Sentiment leads price
- $k < 0$: Price leads sentiment
- Peak at $k = 0$: Contemporaneous

**Hypothesis**: Peak lead time differs between markets.

### 9.4 Herding Intensity Comparison

Compare $\gamma_2$ coefficients:

$$H_0: \gamma_2^{BIST} = \gamma_2^{NASDAQ}$$

**Chow Test**:

$$F = \frac{(SSR_{pooled} - SSR_{BIST} - SSR_{NASDAQ})/k}{(SSR_{BIST} + SSR_{NASDAQ})/(n_{BIST} + n_{NASDAQ} - 2k)}$$

### 9.5 Regime-Dependent Dynamics

Markets may behave differently in bull vs. bear regimes.

**Markov Regime-Switching Model**:

$$R_t = \mu_{S_t} + \phi_{S_t} R_{t-1} + \sigma_{S_t} \epsilon_t, \quad S_t \in \{Bull, Bear\}$$

$$P(S_t = j | S_{t-1} = i) = p_{ij}$$

**Extension with Sentiment**:

$$P(S_t = Bull | S_{t-1}, \text{Sentiment}_{t}) = \frac{1}{1 + \exp(-(\alpha + \beta \cdot \text{Sentiment}_t))}$$

---

## 10. Evaluation Metrics and Statistical Validation

### 10.1 Classification Metrics

For trend prediction $\hat{y} \in \{Up, Down, Neutral\}$:

**Accuracy**:

$$\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision, Recall, F1** (per class):

$$P_c = \frac{TP_c}{TP_c + FP_c}, \quad R_c = \frac{TP_c}{TP_c + FN_c}$$

$$F1_c = \frac{2 P_c R_c}{P_c + R_c}$$

**Macro-F1**: $\frac{1}{C}\sum_{c} F1_c$

**Matthews Correlation Coefficient** (handles imbalance):

$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 10.2 Regression Metrics

If predicting returns $\hat{R}$:

**Mean Squared Error**:

$$MSE = \frac{1}{N} \sum_{t} (R_t - \hat{R}_t)^2$$

**Mean Absolute Error**:

$$MAE = \frac{1}{N} \sum_{t} |R_t - \hat{R}_t|$$

**R-squared**:

$$R^2 = 1 - \frac{\sum_t (R_t - \hat{R}_t)^2}{\sum_t (R_t - \bar{R})^2}$$

### 10.3 Financial Performance Metrics

Trading strategy evaluation:

**Sharpe Ratio**:

$$SR = \frac{E[R_p] - R_f}{\sigma_p}$$

**Maximum Drawdown**:

$$MDD = \max_{t} \left(\max_{s \leq t} V_s - V_t\right)$$

**Calmar Ratio**:

$$CR = \frac{\text{Annual Return}}{|MDD|}$$

**Win Rate**:

$$WR = \frac{\text{Profitable Trades}}{\text{Total Trades}}$$

### 10.4 Statistical Significance

**Diebold-Mariano Test** (comparing forecasts):

$$DM = \frac{\bar{d}}{\hat{\sigma}_d / \sqrt{T}}$$

where $d_t = L(\epsilon_1, t) - L(\epsilon_2, t)$ is the loss differential.

**Bootstrap Confidence Intervals**:

1. Resample predictions with replacement
2. Compute metric on each bootstrap sample
3. Use percentiles as confidence bounds

### 10.5 Walk-Forward Validation

For time series, use expanding or sliding windows:

**Expanding Window**:
- Train on $[1, T_1]$, test on $[T_1+1, T_1+h]$
- Train on $[1, T_1+h]$, test on $[T_1+h+1, T_1+2h]$
- ...

**Sliding Window**:
- Train on $[1, T_1]$, test on $[T_1+1, T_1+h]$
- Train on $[h+1, T_1+h]$, test on $[T_1+h+1, T_1+2h]$
- ...

**Combinatorial Purged Cross-Validation** (de Prado):
Ensures no data leakage across folds by purging overlapping periods.

### 10.6 Ablation Studies

Systematically remove components to measure contribution:

| Model Variant | Accuracy | Sharpe |
|---------------|----------|--------|
| Full model | $A_0$ | $S_0$ |
| Without sentiment | $A_1$ | $S_1$ |
| Without herding correction | $A_2$ | $S_2$ |
| Without attention | $A_3$ | $S_3$ |
| Price only (baseline) | $A_4$ | $S_4$ |

**Significance Test**:

$$t = \frac{A_0 - A_i}{\sqrt{\text{Var}(A_0 - A_i)/n}}$$

---

## Appendix A: Key Notation Reference

| Symbol | Description |
|--------|-------------|
| $P_t$ | Price at time $t$ |
| $R_t$ | Return at time $t$: $(P_t - P_{t-1})/P_{t-1}$ |
| $S_t$ | Aggregated sentiment at time $t$ |
| $D_t$ | Document set for day $t$ |
| $s(d)$ | Sentiment score for document $d$ |
| $h_t$ | Hidden state at time $t$ |
| $c_t$ | Cell state (LSTM) at time $t$ |
| $\sigma(\cdot)$ | Sigmoid function |
| $\odot$ | Element-wise multiplication |
| $CSAD_t$ | Cross-sectional absolute deviation |
| $DI_t$ | Divergence index |
| $SMFI_t$ | Smart money flow index |

---

## Appendix B: Recommended Research Directions

1. **Uncertainty Quantification**: Implement Monte Carlo Dropout or ensemble methods to provide confidence intervals on predictions.

2. **Explainability**: Use attention visualization and LIME/SHAP to understand which tweets drive predictions.

3. **Real-Time Adaptation**: Online learning to adapt to regime changes without full retraining.

4. **Alternative Data**: Extend to Google Trends, options flow, dark pool data.

5. **Multi-Asset Modeling**: Joint prediction of correlated assets using graph neural networks.

---

## References

1. De Long, J. B., Shleifer, A., Summers, L. H., & Waldmann, R. J. (1990). Noise trader risk in financial markets. *Journal of Political Economy*, 98(4), 703-738.

2. Lo, A. W. (2004). The adaptive markets hypothesis. *Journal of Portfolio Management*, 30(5), 15-29.

3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.

4. Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*.

5. Chang, E. C., Cheng, J. W., & Khorana, A. (2000). An examination of herd behavior in equity markets: An international perspective. *Journal of Banking & Finance*, 24(10), 1651-1679.

6. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

7. Antweiler, W., & Frank, M. Z. (2004). Is all that talk just noise? The information content of internet stock message boards. *Journal of Finance*, 59(3), 1259-1294.

8. Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance*, 62(3), 1139-1168.
