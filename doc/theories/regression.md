Regression
==================

Assume that we have dataset :math:`\mathcal{D}={(\mathbf{x}_1, y_1,...,(\mathbf{x}_n,y_n)}` where :math:`\mathbf{x}_i\in\mathbb{R}^m` and :math:`y_i\in\mathbb{R}`, where :math:`\mathbf{x}_i` consists of :math:`m` measurements or attributes associated with the :math:`i`-th observation and :math:`y_i` is the :math:`i`-th target variable which is some real number. Then we are interested in learning some functional map that maps from our observation space to our target space i.e. :math:`\mathcal{X}\rightarrow\mathcal{Y}`. There are a multitude of assumptions we could make about defining the best :math:`f` but the most popular is to assume :math:`f` is the curve of best fit through our data and the noise present is Gaussian. Mathematically we can define the functional relationship to be

.. math::
    \begin{equation}
        y_i = f(\mathbf{x}_i;\boldsymbol{\theta}) + \epsilon_i, \qquad \epsilon_i \overset{\text{iid}}{\sim}\mathcal{N}(0,\sigma^2),
    \end{equation}


where :math:`\boldsymbol{\theta}` are the parameters for our functional map :math:`f` and :math:`\epsilon_i` is some random Gaussian noise. The reason why we can assume that noise is present is because we often may not have all the information in :math:`\mathbf{X}` to construct :math:`\mathbf{y}` or our proposed function :math:`f` may be simpler than the true function :math:`f^{*}` and so we are finding the best simpler model that fits the more complicated dataset.

From Eq. (1), we can say that our predictor for :math:`y_i` is :math:`\mathbb{E}[f(\mathbf{x}_i;\boldsymbol{\theta}) + \epsilon_i] = \mathbb{E}[f(\mathbf{x}_i;\boldsymbol{\theta})]`. For notational convenience let us denote :math:`\mu(\mathbf{x}_i;\boldsymbol{\theta}) = \mathbb{E}[f(\mathbf{x}_i;\boldsymbol{\theta})]`, and assume that our function :math:`f` is deterministic i.e. a strict one to one map. We can then say that

.. math::
    \begin{equation}
        y_i|\mathbf{x}_i,\boldsymbol{\theta},\sigma^2 \overset{\text{iid}}{\sim}\mathcal{N}(\mu(\mathbf{x}_i; \boldsymbol{\theta}),\sigma^2),
    \end{equation}

that is, our target variable given an associated distribution follows an independent and identically distributed Gaussian with mean :math:`\mu(\mathbf{x}_i,\boldsymbol{\theta})` and variance :math:`\sigma^2`. To visualise the relationships between the variables we have the below.


.. figure:: /_static/theories/regression/regression.png
    :scale: 80 %
    :align: center
    
    Graphical model representation of the regression problem. Dotted circles are the unknowns of interest, shaded circles are the data, and regular circles are functions.

Writing the conditional probability density of Eq. (2) we yield

.. math::
    \begin{equation}
        p(y_i|\mathbf{x}_i,\boldsymbol{\theta},\sigma^2) = (2\pi\sigma^2)^{-\frac{1}{2}}\exp\bigg(-\frac{1}{2\sigma^2}\big(y_i - \mu(\mathbf{x}_i;\boldsymbol{\theta})\big)^2\bigg). \label{eq:normal-pdf}
    \end{equation}

Assuming that each of the :math:`n` observations are independent, we can state

.. math::
    :nowrap:
        
    \begin{align}
         p(\mathbf{y}|\mathbf{X},\boldsymbol{\theta},\sigma^2) &= \prod_{i=1}^n p(y_i|\mathbf{x}_i,\boldsymbol{\theta},\sigma^2), \label{eq:likelihood}\\
        &= (2\pi\sigma^2)^{-\frac{n}{2}}\exp\bigg(-\frac{1}{2\sigma^2}\sum_{i=1}^n\big(y_i - \mu(\mathbf{x}_i;\boldsymbol{\theta})\big)^2\bigg), \\
        \log p(\mathbf{y}|\mathbf{X},\boldsymbol{\theta},\sigma^2) &= -\frac{n}{2}\log 2\pi\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n\big(y_i - \mu(\mathbf{x}_i;\boldsymbol{\theta})\big)^2, \label{eq:sum}\\
        &= -\frac{n}{2}\log 2\pi\sigma^2 - \frac{1}{2\sigma^2}\big(\mathbf{y} - \boldsymbol{\mu}(\mathbf{X};\boldsymbol{\theta})\big)^\text{T}\big(\mathbf{y} - \boldsymbol{\mu}(\mathbf{X};\boldsymbol{\theta})\big),  \label{eq:transpose}\\
        &= -\frac{n}{2}\log 2\pi\sigma^2 - \frac{1}{2\sigma^2}||\mathbf{y} - \boldsymbol{\mu}(\mathbf{X};\boldsymbol{\theta})||_2^2. \label{eq:norm}
    \end{align}

.. note ::
    
    .. raw:: html
    
        <details>
        <summary>The notation used in Eqs. (6-8) all mean the same thing.</summary>
        <br>

    For simplicity, let :math:`\mathbf{z} = \mathbf{y} - \boldsymbol{\mu}(\mathbf{X};\boldsymbol{\theta})` and to simplify further assume that :math:`\mathbf{z}\in\mathbb{R}^{3}` i.e. :math:`\mathbf{z} = [z_1,z_2,z_3]`, then:
    
    .. math ::
        :nowrap:

        \begin{align*}
            \mathbf{z}^\text{T}\mathbf{z} &= \begin{bmatrix} z_1 & z_2 & z_3\end{bmatrix} \begin{bmatrix}z_1\\z_2\\z_3\end{bmatrix},\\
            &= z_1^2 + z_2^2 + z_3^2, \\
            &= \sum_{i = 1}^3 z_i^2.
        \end{align*}
    
    The jump from Eq. :math:`\eqref{eq:sum}` to Eq. :math:`\eqref{eq:transpose}` is now explained. Eq. :math:`\eqref{eq:norm}` comes from the definition of a vector norm.

    .. math ::
        :nowrap:
    
        \begin{align*}
            ||\mathbf{z}||_p^q :&= \bigg[\sum_{i = 1}^3 |z_i|^p\bigg]^{q / p}, \\
            ||\mathbf{z}||_2^2 &= \bigg[\sum_{i=1}^3 |z_i|^2\bigg]^{2 / 2} (\text{setting } p = q = 2),\\
            &= \sum_{i=1}^3 z_i^2.
        \end{align*}

    .. raw:: html

        </details>


The probability distribution :math:`p(\mathbf{y}_i\mid\mathbf{x}_i,\boldsymbol{\theta},\sigma^2)` stated in Eq. :math:`\eqref{eq:normal-pdf}` indicates a distribution over :math:`\mathcal{Y}` with the most probable value to be :math:`\mu(\mathbf{x}_i;\boldsymbol{\theta})`. Likewise, for the entire dataset, the most likely set of values associated with :math:`\mathbf{X}` is :math:`\boldsymbol{\mu}(\mathbf{X},\boldsymbol{\theta})`. In plain english, the left hand side of Eq. :math:`\eqref{eq:likelihood}` is the likelihood (or probability) that we would observe :math:`\mathbf{y}` if we started from :math:`\mathbf{X}` and used Eq. (1) to generate :math:`\mathbf{y}`. If the probability is low then we have a poor functional map :math:`f` and similarly, if we the probability is high then we have a good functional map :math:`f`. As the function log just converts the probability space :math:`(0,1)` to the log probability space :math:`(-\infty,0)`, we can choose to maximise the log probability instead. This is often done in practice as the resulting math is much easier and maximising the log probability also maximises the probability.

To maximise Eqs. (:math:`\ref{eq:sum}`-:math:`\ref{eq:norm}`) (all the same expression) with respect to :math:`\boldsymbol{\theta}`, we can consider all other variables constants and simply ignore them for the optimisation step. We can see that maximising the log probability is equivalent to minimising

.. math::
    \begin{equation}
        \mathcal{L}(\boldsymbol{\theta};\mathbf{X},\mathbf{y}) := ||\mathbf{y} - \boldsymbol{\mu}(\mathbf{X};\boldsymbol{\theta})||_2^2.
    \end{equation}

To minimise :math:`\mathcal{L}(\boldsymbol{\theta};\mathbf{X},\mathbf{y})`, we can consider the Jacobian (first derivative) and perform gradient descent,

.. math::
    \begin{equation}
        \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta};\mathbf{X},\mathbf{y}) = -2\nabla_{\boldsymbol{\theta}} [\boldsymbol{\mu}(\mathbf{X};\boldsymbol{\theta})]^\text{T}\odot\big(\mathbf{y} - \boldsymbol{\mu}(\mathbf{X};\boldsymbol{\theta})\big), \label{eq:grad}
    \end{equation}

where :math:`\odot` denotes element-wise multiplication.

.. note::
    
    .. raw:: html
        
        <details>
        <summary>Differentiating the loss function.</summary><br>

    :math:`\nabla_\theta \mathcal{L}(\theta;\mathbf{X},\mathbf{y})` means the vector differential of :math:`\mathcal{L}(\theta;\mathbf{X},\mathbf{y})` with respect to :math:`\theta`. Suppose that there are three parameters :math:`\theta_1,\ \theta_2 \text{ and } \theta_3`, and we want to find the differential of :math:`\mathcal{L}((\theta_1, \theta_2, \theta_3);\mathbf{z}) = ||\mathbf{z}(\theta_1,\theta_2,\theta_3)||_2^2` with respect to :math:`\theta_1` where :math:`\mathbf{z} = [z_1,z_2,z_3]` then:

    .. math::
        :nowrap:

        \begin{align*}
            \frac{\partial \mathcal{L}}{\partial \theta_1} &= \frac{\partial}{\partial \theta_1}\bigg[||\mathbf{z}||_2^2\bigg],\\
            &= \frac{\partial}{\partial \theta_1}\bigg[\sum_{i=1}^3 z_i^2\bigg], \\
            &= \sum_{i=1}^3 \frac{\partial}{\partial \theta_1} [z_i^2], \\
            &= \sum_{i=1}^3 2\frac{\partial}{\partial \theta_1}[z_i] z_i,\\
            &= 2\frac{\partial \mathbf{z}}{\partial \theta_1}^\text{T}\mathbf{z},\\
            \Rightarrow \nabla_{\theta} [\mathcal{L}] &= \begin{bmatrix}\frac{\partial \mathcal{L}}{\partial \theta_1} \\ \frac{\partial \mathcal{L}}{\partial \theta_2} \\ \frac{\partial \mathcal{L}}{\partial \theta_3}\end{bmatrix},\\
            &= 2\begin{bmatrix}\frac{\partial \mathbf{z}^\text{T}}{\theta_1}\mathbf{z} \\ \frac{\partial \mathbf{z}^\text{T}}{\theta_2}\mathbf{z} \\ \frac{\partial \mathbf{z}^\text{T}}{\theta_3}\mathbf{z}\end{bmatrix}, \\
            &= 2\nabla_{\theta} [\mathbf{z}]^\text{T} \odot \mathbf{z},
        \end{align*}

    where :math:`\cdot` denotes element-wise multiplication.

If our functional map :math:`f` is linear, we can solve Eq. :math:`\eqref{eq:grad}` directly and this is called |linear-regression|, if it is non-linear, we typically cannot solve it and have to resort to gradient descent. For all deterministic functional maps :math:`f`, this underlying theory will apply. If the functional map :math:`f` is stochastic i.e. has an underlying distribution a more general set of assumptions will have to be made around the variance :math:`\sigma^2` attributed to the distribution of :math:`\mathbf{y}` as there will be covariances introduced. See the |gaussian-process| as an example.

.. |linear-regression| raw:: html

    <a href="../how-to-ml/supervised-learning/linear-regression.html" target="_blank">linear regression</a>

.. |gaussian-process| raw:: html

    <a href="../how-to-ml/supervised-learning/gaussian-process.html" target="_blank">gaussian process</a>


