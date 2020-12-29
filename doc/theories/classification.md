Classification
======================


The objective here is to predict a class label for an observation in a supervised setting.


Binary Classification
---------------------

Similar to the |regression| case, assume we have a dataset :math:`\mathcal{D} = \{(\mathbf{x}_1,y_1),...,(\mathbf{x}_n,y_n)\}` where :math:`\mathbf{x}_i\in\mathbb{R}^m` and :math:`y_i\in\{0,1\}` where :math:`\mathbf{x}_i` consists of :math:`m` measurement or attributes associated with the :math:`i`-th observation and :math:`y_i` is the :math:`i`-th target variable which indicates if it is present (1) or absent (0). Then we are interested in learning some functional map that maps from our observation space to our target space i.e. :math:`f : \mathcal{X} \rightarrow \mathcal{Y}`. At this stage it is more appropriate to think of :math:`\mathbf{y}` as a probability space which we denote as :math:`\mathbf{p}` from now on and our model prediction :math:`\mathbf{q}(\mathbf{X};\boldsymbol{\theta}):=f(\mathbf{X};\boldsymbol{\theta})`. Then by examining the Bernoulli likelihood

.. math::
    :nowrap:

    \begin{align}
        p(p_i|\mathbf{x}_i,\boldsymbol{\theta}) &= q_i^{p_i}(1 - q_i)^{1-p_i},\qquad q_i = q(\mathbf{x}_i;\boldsymbol{\theta}),\\
        p(\mathbf{p}|\mathbf{X},\boldsymbol{\theta}) &= \prod_{i=1}^n q_i^{p_i}(1 - q_i)^{1-p_i},\\
        \log p(\mathbf{p}|\mathbf{X},\boldsymbol{\theta}) &= \sum_{i = 1}^n p_i\log q_i + (1 - p_i)\log (1 - q_i). \label{eq:binary-entropy}
    \end{align}


We examine the Bernoulli likelihood because if our model was completely correct we yield a probability of 1. As we wish to maximise the log probability in Eq. :math:`\eqref{eq:binary-entropy}`, we can compute the gradients of Eq. :math:`\eqref{eq:binary-entropy}` and update :math:`\boldsymbol{\theta}` using gradient ascent. It is popular to assume that the function :math:`q(\mathbf{x}_i;\boldsymbol{\theta}) = q_i = \phi(z_i)` where :math:`\phi` is the logistic sigmoid function, :math:`z_i = z(\mathbf{x}_i;\boldsymbol{\theta})`, and :math:`z` is a function that maps our observations into `log-odds <https://en.wikipedia.org/wiki/Logit>`_ space. The logistic sigmoid function is defined by

.. math::
    :nowrap:

    \begin{align}
        \phi(z) &= \frac{1}{1 + \exp(-z)} = \frac{\exp(z)}{1 + \exp(z)},\\\nonumber\\
        \frac{\text{d} \phi(z)}{\text{d}z} &= \frac{\exp(-z)}{\big(1 + \exp(-z)\big)^2} = \frac{1}{1 + \exp(-z)} \frac{\exp(-z)}{1 + \exp(-z)},\\
        &=  \frac{1}{1 + \exp(-z)} \frac{1 + \exp(-z) - 1}{1 + \exp(-z)} = \frac{1}{1 + \exp(-z)}\bigg(1 - \frac{1}{1 + \exp(-z)}\bigg),\\
        &= \phi(z)\big(1 - \phi(z)\big). \label{eq:phi-chain}
    \end{align}

By substituting Eq. :math:`\eqref{eq:phi-chain}` into Eq. :math:`\eqref{eq:binary-entropy}` and collecting terms together we yield

.. math::
    :nowrap:
    
    \begin{align}
        \log p(\mathbf{p}|\mathbf{X},\boldsymbol{\theta}) &= \sum_{i = 1}^n \log (1 - q_i) + p_i\log\frac{q_i}{1 - q_i},\\
        &= \sum_{i=1}^n \log \big(1 - \phi(z_i)\big) + p_i \log \exp(z_i),\\
        &= \sum_{i=1}^n \log \big(1 - \phi(z_i)\big) + p_i z_i,\\\nonumber\\
        \nabla_{\boldsymbol{\theta}}[\log p(\mathbf{p}|\mathbf{X},\boldsymbol{\theta})] &= \sum_{i=1}^n -\frac{\phi(z_i)\big(1 - \phi(z_i)\big) \nabla_{\boldsymbol{\theta}} [z_i]}{1 - \phi(z_i)} + p_i \nabla_{\boldsymbol{\theta}} [z_i]\\
        &= \sum_{i=1}^n \big(p_i - \phi(z_i)\big)\nabla_{\boldsymbol{\theta}} [z_i],\\
        &= \nabla_{\boldsymbol{\theta}}[\mathbf{z}]^\text{T}(\mathbf{p} - \mathbf{q}). \label{eq:binary}
    \end{align}

By examining the final expression for the gradient, we can see a similarity between how we would update :math:`\boldsymbol{\theta}` for the regression and classification cases.

Multilabel Classification
-------------------------

Similar to the above binary case, assume we have a dataset :math:`\mathcal{D} = \{(\mathbf{x}_1,\mathbf{y}_1),...,(\mathbf{x}_n,\mathbf{y}_n)\}` where :math:`\mathbf{x}_i\in\mathbb{R}^m` and :math:`\mathbf{y}_i\in\{0,1\}^c`. What has changed here is that we are trying to classify each observation into a set of :math:`c` classes and each observation can belong to up to :math:`c` classes. For notation, let :math:`\mathbf{P} = \mathbf{Y}` and :math:`\mathbf{Q} = \mathbf{Q}(\mathbf{X};\boldsymbol{\theta}) = \phi(\mathbf{Z})`, and :math:`z_{ij} = z(\mathbf{x_{i}};\boldsymbol{\theta})_j` where :math:`z` is a function that maps an observation vector of size :math:`m` to a log-odds vector of size :math:`c`. The multivariate generalisation of the Bernoullli distribution is known as the Categorical or Multinoulli distribution. The likelihood has a similar form

.. math::
    :nowrap:

    \begin{align}
        p(\mathbf{p}_i|\mathbf{x}_i,\boldsymbol{\theta}) &= \prod_{j=1}^c q_{ij}^{p_{ij}}\big(1 - q_{ij}\big)^{1 - p_{ij}},\\
        p(\mathbf{P}|\mathbf{X},\boldsymbol{\theta}) &= \prod_{i=1}^n\prod_{j=1}^c q_{ij}^{p_{ij}}\big(1 - q_{ij}\big)^{1 - p_{ij}},\\
        \log p(\mathbf{P}|\mathbf{X},\boldsymbol{\theta}) &= \sum_{i=1}^n\sum_{j=1}^c p_{ij} \log q_{ij} + (1 - p_{ij}) \log \big(1 - q_{ij}\big),\\
        &=  \sum_{i=1}^n\sum_{j=1}^c \log \big(1 - \phi(z_{ij})\big) + p_{ij}z_{ij},\\\nonumber\\
        \nabla_{\boldsymbol{\theta}} [\log p(\mathbf{P}|\mathbf{X},\boldsymbol{\theta})] &= \sum_{i=1}^n\sum_{j=1}^c -\frac{\phi(z_{ij})\big(1 - \phi(z_{ij})\big) \nabla_{\boldsymbol{\theta}} [z_{ij}]}{1 - \phi(z_{ij})} + p_{ij} \nabla_{\boldsymbol{\theta}} [z_{ij}], \label{eq:chain}\\
        &= \sum_{i=1}^n\sum_{j=1}^c (p_{ij} - q_{ij}) \nabla_{\boldsymbol{\theta}}[z_{ij}],\\
        &= \nabla_{\boldsymbol{\theta}} [\mathbf{Z}]^\text{T} (\mathbf{P} - \mathbf{Q}), \label{eq:multilabel}
    \end{align}

where we use the chain rule for the log in Eq. :math:`\eqref{eq:chain}` as well as the result from Eq. :math:`\eqref{eq:phi-chain}`. Note the similar form between Eq. :math:`\eqref{eq:multilabel}` and Eq. :math:`\eqref{eq:binary}`!

Multiclass Classification
-------------------------

Similar to the above multilabel case, assume we have a dataset :math:`\mathcal{D} = \{(\mathbf{x}_1,\mathbf{y}_1),...,(\mathbf{x}_n,\mathbf{y}_n)\}` where :math:`\mathbf{x}_i\in\mathbb{R}^m` and :math:`\mathbf{y}_i\in\{0,1\}^c` where :math:`\sum_{j=1}^c y_{ij} = 1\ \forall\ i\in\{1,...,n\}`. What has changed here is that we are trying to classify each observation into one of :math:`c` classes and each observation strictly belongs to a single class. For notation, we will keep the majority the same as the multilabel case - the main difference is changing the function :math:`\phi`. Previously :math:`\phi` was the logistic sigmoid function defined in Eq. :math:`\eqref{eq:phi-chain}`, but since for a vector of values :math:`\mathbf{q}_i = \phi(\mathbf{z}_i)`, we cannot guarantee that :math:`\sum_{j=1}^c q_{ij} = 1` which is a requirement as an observation can only belong to a single class so we are interested in quantifying a probability distribution over the classes for each observation. The function we use in place of the logistic sigmoid is known as the softmax function which we denote from this point on as

.. math::
    :nowrap:

    \begin{align}
        \phi(\mathbf{z})_i &= \frac{\exp(z_{i})}{\sum_{k=1}^c \exp(z_{k})},\\\nonumber\\
        \frac{\partial \phi(\mathbf{z})_i}{\partial z_j} &= \frac{\sum_{k = 1}^c \exp(z_k)\frac{\partial}{\partial z_j}\big[\exp(z_i)\big] - \exp(z_i)\frac{\partial}{\partial z_j}\big[\sum_{k=1}^c \exp(z_k)\big]}{\big(\sum_{k = 1}^c \exp(z_k)\big)^2}, \label{eq:quotient}\\
        &= \frac{\sum_{k=1}^c \exp(z_k)\exp(z_j)\delta_{ij} - \exp(z_i)\exp(z_j)}{\big(\sum_{k = 1}^c \exp(z_k)\big)^2},\\
        &= \frac{\exp(z_j)\delta_{ij}}{\sum_{k=1}^c \exp(z_k)} - \frac{\exp(z_i)}{\sum_{k=1}^c \exp(z_k)}\frac{\exp(z_j)}{\sum_{k=1}^c \exp(z_k)},\\
        &= \big(\delta_{ij} - \phi(\mathbf{z})_i\big)\phi(\mathbf{z})_j,\label{eq:softmax-chain}
    \end{align}

where :math:`\delta_{ij} = 1` if :math:`i=j` and :math:`\delta_{ij} = 0` if :math:`i\neq j` (this is known as the Kronecker delta function). Eq. :math:`\eqref{eq:quotient}` can be explained by using the `quotient rule <https://en.wikipedia.org/wiki/Quotient_rule>`_. The log likelihood with a softmax :math:`\phi` definition is then

.. math::
    :nowrap:

    \begin{align}
        \log p(\mathbf{P}|\mathbf{X},\boldsymbol{\theta}) &= \sum_{i=1}^n\sum_{j=1}^c p_{ij} \log q_{ij},\\
        &= \sum_{i=1}^n\sum_{j=1}^c p_{ij} \log \phi(\mathbf{z}_i)_j,\\\nonumber\\
        \nabla_{\boldsymbol{\theta}} [\log p(\mathbf{P}|\mathbf{X},\boldsymbol{\theta})]_k &= \sum_{i=1}^n\sum_{j=1}^c p_{ij} \frac{\big(\delta_{jk} - \phi(\mathbf{z}_i)_k\big)\phi(\mathbf{z}_i)_j}{\phi(\mathbf{z}_i)_j} \nabla_{\boldsymbol{\theta}} [z_{ik}], \label{eq:chain2}\\
        &= \sum_{i=1}^n\sum_{j=1}^c p_{ij} (\delta_{jk} - q_{ik}) \nabla_{\boldsymbol{\theta}} [z_{ik}],\\
        &= \sum_{i = 1}^n\nabla_{\boldsymbol{\theta}} [z_{ik}]\bigg(p_{ik} - q_{ik}\sum_{j=1}^c p_{ij}\bigg),\\
        &= \sum_{i = 1}^n\nabla_{\boldsymbol{\theta}} [z_{ik}]\bigg(p_{ik} - q_{ik}\bigg),\\\nonumber\\
        \nabla_{\boldsymbol{\theta}} [\log p(\mathbf{P}|\mathbf{X},\boldsymbol{\theta})] &= \nabla_{\boldsymbol{\theta}} [\mathbf{Z}]^\text{T}(\mathbf{P} - \mathbf{Q}), \label{eq:multiclass}
    \end{align}

where we use the chain rule for Eq. :math:`\eqref{eq:chain2}` and have the differential of log as well as the result found in Eq. :math:`\eqref{eq:softmax-chain}`, and since each :math:`\mathbf{p}_{i}` consists of a single 1 and the rest 0 we know that :math:`\sum_{j=1}^c p_{ij} = 1`. Note the similar form between Eq. :math:`\eqref{eq:multiclass}` and Eq. :math:`\eqref{eq:multilabel}`!


Summary
-------

Regardless if the task is a binary, multilabel, or multiclass classification, the update rule is the same for any function :math:`z(\mathbf{X};\boldsymbol{\theta})` (assuming we use the same definitions of :math:`\phi` in each case). See |logistic-regression| for the simple case where :math:`z` is a linear function of the data.

.. |regression| raw:: html

    <a href="regression.html" target="_blank">regression</a>

.. |logistic-regression| raw:: html

    <a href="../how-to-ml/supervised-learning/logistic-regression.html" target="_blank">logistic regression</a>


