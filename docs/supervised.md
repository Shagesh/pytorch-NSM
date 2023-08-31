# Supervised learning with similarity matching

Similarity matching was initially introduced as an [unsupervised method](similarity.md). A form of supervision can be introduced by using an additional term in the optimization objective:[^1]

$$ \min_{\mathbf y} \frac 1{T^2}\sum_{t, t'=0}^T \bigl\lvert W(\mathbf x_t) W(\mathbf x_{t'}) + \alpha \mathbf z_t \mathbf z_{t'} - \mathbf y_t \mathbf y_{t'}\bigr\rvert^2\,,$$

where $\mathbf z$ are the labels. The coefficient $\alpha$ can be used to tune how important the inputs or the labels are for the output of the circuit. This is a more symmetric form of supervision that is more similar to self-supervised learning. This may be a feature of the model, as labeled data is generally harder to find, both in artificial and in biological systems.

## Multiple input pathways
The supervision is introduced above as a term in the objective function similar to the input. We can generalize this to arbitrarily many input modalities:

$$ \min_{\mathbf y} \frac 1{T^2}\sum_{t, t'=0}^T \Bigl\lvert W_1(\mathbf x^1_t) W_1(\mathbf x^1_{t'}) + \dotsb + W_k(\mathbf x^k_t) W_k(\mathbf x^k_{t'}) - \mathbf y_t \mathbf y_{t'}\Bigr\rvert^2\,.$$

This models a multi-view circuit, in which several different types of inputs are combined to generate the output.

[^1]: Bahroun, Y., Sridharan, S., Acharya, A., Chklovskii, D. B., & Sengupta, A. M. (2023). Unlocking the Potential of Similarity Matching: Scalability, Supervision and Pre-training. <http://arxiv.org/abs/2308.02427>