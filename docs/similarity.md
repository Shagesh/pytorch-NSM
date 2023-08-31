# Similarity matching

## Basic form
Similarity matching was introduced in 2014 as an algorithm for sparse dictionary learning that can be implemented in a biologically plausible manner.[^1] The key innovation was to use an objective function that focuses on the similarity between inputs $\mathbf x_t$ and outputs $\mathbf y_t$, akin to multi-dimensional scaling:

$$ \min_{\mathbf y} \frac 1{T^2}\sum_{t, t'=0}^T \lvert \mathbf x_t \mathbf x_{t'} - \mathbf y_t \mathbf y_{t'}\rvert^2\,.$$

In this form the optimization is far from biological plausibility, since the objective contains a number of terms that is quadratic in the number of samples $T$. However, this can be turned into an online algorithm based on Hebbian learning by using a trick involving auxiliary variables.[^2] We start by expanding the square

$$ \min_{\mathbf y} \frac 1{T^2}\left\{\sum_t \mathbf y_t^\top \left[\sum_{t'} \mathbf y_{t'} \mathbf y_{t'}^\top\right] \mathbf y_t - 2 \sum_t \mathbf y_t^\top \left[\sum_{t'} \mathbf y_{t'} \mathbf x_{t'}^\top\right] \mathbf x_t\right\}\,,$$

where we ignored the term quadratic in $\mathbf x$ which is independent of the optimization variables. We define the auxiliary variables

$$\begin{split}
W &= \frac 1T \sum_{t'} \mathbf y_{t'} \mathbf x_{t'}^\top\,,\\
M &= \frac 1T \sum_{t'} \mathbf y_{t'} \mathbf y_{t'}^\top\,,
\end{split}
$$

and note that these definitions can themselves be obtained from an optimization:

$$
\min_{\mathbf y}\min_{W}\max_{M}\biggl\{ \frac 2T \sum_t \mathbf y_t^\top M \mathbf y_t - \frac 4T \sum_t \mathbf y_t^\top W \mathbf x_t + 2 \mathop{\text{Tr}} W^\top W - \mathop{\text{Tr}} M^\top M \biggr\}\,.
$$

The next step is noticing that the order of optimization can be swapped so that the minimization over $\mathbf y$ happens first.[^2],

$$
\min_{W}\max_{M}\min_{\mathbf y}\biggl\{ \frac 2T \sum_t \mathbf y_t^\top M \mathbf y_t - \frac 4T \sum_t \mathbf y_t^\top W \mathbf x_t + 2 \mathop{\text{Tr}} W^\top W - \mathop{\text{Tr}} M^\top M \biggr\}\,.
$$

Finally, an online algorithm is obtained by writing the objective as a sum over time $t$ so that each term can be optimized independently[^2]. The optimization over the latest sample $\mathbf y_t$ reduces to

$$\min_{\mathbf y_t} \Bigl[-4 \mathbf x_t^\top W^\top \mathbf y_t + 2 \mathbf y_t^\top M \mathbf y_t\Bigr]\,,$$

while the optimization over $W$ and $M$ can be performed via gradient descent, resulting in updates of the form

$$
\begin{split}
\Delta W &= 2 \eta \bigl[\mathbf y_t \mathbf x_t^\top - W] \,,\\
\Delta M &= \frac {\eta} {\tau} \bigl[\mathbf y_t \mathbf y_t^\top - M]\,,
\end{split}
$$

where $\eta$ is a learning rate and $\tau$ is a ratio of learning rates.

![Biological circuit](img/basic_circuit.png)

## Non-negative similarity matching (NSM)
In its linear form derived above, the similarity matching algorithm simply converges to principal subspace projection (PSP).[^3] More diverse functions, ranging from sparse coding[^1] to manifold tiling[^4], can be obtained by adding a non-negativity constraint to the optimization:

$$\min_{\mathbf y_t\ge 0} \Bigl[-4 \mathbf x_t^\top W^\top \mathbf y_t + 2 \mathbf y_t^\top M \mathbf y_t\Bigr]\,,$$

where the non-negativity is to be understood component-wise.

Note that the non-negativity only affects the fast dynamics that sets the output value $\mathbf y$. The synaptic plasticity rules stay the same.

For more details, see [The search for biologically plausible neural computation: A similarity-based approach](http://www.offconvex.org/2018/12/03/MityaNN2/).

## General encoder
We can use an arbitrary encoder on the inputs, turning the optimization into[^5]

$$ \min_{\mathbf y} \frac 1{T^2}\sum_{t, t'=0}^T \lvert W(\mathbf x_t) W(\mathbf x_{t'}) - \mathbf y_t \mathbf y_{t'}\rvert^2\,.$$

Here $W$ can be any function—for instance, a deep neural network.

[^1]: Hu, T., Pehlevan, C., & Chklovskii, D. B. (2014). *A Hebbian/Anti-Hebbian network for online sparse dictionary learning derived from symmetric matrix factorization.* 2014 48th Asilomar Conference on Signals, Systems and Computers, 613–619. <https://doi.org/10.1109/ACSSC.2014.7094519>

[^2]: Pehlevan, C., Sengupta, A. M., & Chklovskii, D. B. (2018). *Why Do Similarity Matching Objectives Lead to Hebbian/Anti-Hebbian Networks?* Neural Computation, 30, 84–124. <https://doi.org/10.1162/NECO>

[^3]: Pehlevan, C., & Chklovskii, D. B. (2015). *Optimization theory of Hebbian/anti-Hebbian networks for PCA and whitening.* Communication, Control, and Computing (Allerton), 1458–1465. <https://doi.org/10.1109/ALLERTON.2015.7447180>

[^4]: Sengupta, A. M., Tepper, M., Pehlevan, C., Genkin, A., & Chklovskii, D. B. (2018). *Manifold-tiling Localized Receptive Fields are Optimal in Similarity-preserving Neural Networks.* Advances in Neural Information Processing Systems, 7080–7090.

[^5]: Bahroun, Y., Sridharan, S., Acharya, A., Chklovskii, D. B., & Sengupta, A. M. (2023). Unlocking the Potential of Similarity Matching: Scalability, Supervision and Pre-training. <http://arxiv.org/abs/2308.02427>
