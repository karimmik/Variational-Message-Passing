# Variational Message Passing

Below is a simple example of using Variational Message Passing (VMP) to approximate the **mean** and **variance** of an unknown distribution. In this example, the data are generated from a Normal distribution.

## Some Theory
*Original idea by John Winn and Christopher M. Bishop. 2005. Variational Message Passing. J. Mach. Learn. Res. 6 (12/1/2005), 661–694.*

VMP adopts the same basic assumption as mean-field variational inference:

$$
Q(\mathbf{H})=\prod_{i} Q_{i}(\mathbf{H}_{i})
$$

The idea is to factor the hidden variables into disjoint groups. While this factorization is rarely exact, it makes inference tractable by removing otherwise intractable integrals over conditionals—an assumption sufficient to start the procedure.

At each iteration, we update one factor while holding the others fixed (coordinate ascent). After updating a factor, I move on to the next, and repeat:

$$
\ln Q_i^\*(\mathbf H_i)
= \big\langle \ln p(\mathbf H,\mathbf V) \big\rangle_{Q_{\neg i}} + \text{const}
$$

VMP expresses these updates as **local messages** on a factor graph: each update needs only expectations from neighboring nodes (expected natural parameters), which makes the algorithm modular and fast.
## Model used in the code

Assuming data $(x_1,\dots,x_N)$ with $x_n \mid \mu, \sigma \sim \mathcal N(\mu,\, \sigma^2),$
and independent priors $\mu \sim \mathcal N(m_0,\ \beta), \sigma^2 \sim \mathrm{InvGamma}(a,\ b). $

I approximate with the mean-field family

$$
Q(\mu)\,Q(\sigma^2)
= \mathcal N(\mu;\ m,\ \beta) \times \mathrm{InvGamma}(\sigma^2; a, b).
$$

> **Notation.** For the variance node $\sigma^2$ I use the **shape–scale** parameterization.
> Useful expectations:
> $\mathbb E\!\left[\tfrac{1}{\sigma^2}\right]=\frac{a}{b},
> \qquad
> \mathbb E[\ln \sigma^2]=\ln(b)-\psi(a).$

---

## Messages and updates (what the code implements)

### 1) Variance $\sigma^2\to$ data $x_n$

Same for all $n$:

$$\langle \mathbf u_\sigma^2 \rangle=\begin{bmatrix}
\langle 1/\sigma^2\rangle\\\
\langle \ln \sigma^2\rangle
\end{bmatrix}=
\begin{bmatrix}
\alpha/\beta\\\
\ln\beta-\psi(\alpha)
\end{bmatrix}.$$

### 2) Data $x_n \to$ mean $\mu$
Using only $\langle 1/\sigma^2\rangle$:

$$
m_{x\to \sigma^2} =
\begin{bmatrix}
\langle 1/\sigma^2\rangle\,x_i\\\
-\tfrac12\,\langle 1/\sigma^2\rangle
\end{bmatrix}.
$$

Summing over \(n\) and adding the Normal prior on $\mu$ gives canonical parameters

$$
\phi_u =
\begin{bmatrix}
\beta\mu\\\
-\beta/2
\end{bmatrix} +
\sum_{n=1}^N m_{x\to \mu}
$$

$$
\mu=-2 * \phi_u[1]
\qquad
\beta=\phi_u[0] / -2 * \phi_u[1]
$$

### 3) Mean $\mu \to$ data $x_n$
The message needs the moments of $\mu$:

$$
\langle \mu\rangle = m,
\qquad
\langle \mu^2\rangle = m^2 + \beta^{-1}.
$$

### 4) Data $x_n \to$ variance $\sigma^2$
Per datum (in the $[\ln \sigma^2,\ 1/\sigma^2]$ sufficient-stat order):

$$
m_{x\to y} =
\begin{bmatrix}
-\tfrac12\\\
-\tfrac12\,\mathbb E\!\big[(x_i-\mu)^2\big]
\end{bmatrix},
\qquad
\mathbb E\!\big[(x_i-\mu)^2\big]=(x_i-m)^2+\beta^{-1}.
$$

Summing over $n$ combining with the $\mathrm{InvGamma}(\alpha_0,\beta_0)$ prior yields the closed-form update

$$
\alpha = \alpha_0 + \frac{N}{2},
\qquad
\beta = \beta_0 + \frac{1}{2}\sum_{n=1}^{N}\big((x_n-m)^2+\beta^{-1}\big).
$$

### Results and Conclusion

I implemented Variational Message Passing (VMP) for a Gaussian likelihood with
an unknown mean and variance, using a Normal factor for the mean and an
Inverse-Gamma factor for the variance. Under the mean-field assumption
$Q(\mu)Q(\sigma^2)$, each iteration requires only local expectations:

- the mean update uses $\mathbb{E}[1/\sigma^2] = \alpha/\beta$;
- the variance update uses the summed expected squared residuals
  $\sum_n \mathbb{E}[(x_n-\mu)^2] = \sum_n (x_n-m)^2 + N/\beta$.

![Mean approximation](img/output_mean.png)
![Variance approximation](img/output_variance.png)

With sensible priors (weak Normal prior on $\mu$, shape–scale Inv-Gamma for $\sigma^2$
centered near the data variance) the algorithm converges quickly and stably:
$m$ moves from the prior toward the sample mean, while the posterior over
$\sigma^2$ tightens around the empirical variance. Using fixed priors $(\alpha_0,\beta_0)$
at every iteration (rather than accumulating $\alpha$) and maintaining
the correct message signs/order $[\ln \sigma^2\,1/\sigma^2]$ prevents numerical issues
(NaNs, negative scales). Optional damping further stabilizes updates on
noisy datasets.

Overall, VMP reproduces the closed-form conjugate updates while exposing
them as modular “messages,” making the approach easy to extend to larger
graphs (e.g., hierarchical means, mixture models).


