Logistic Regression
###################

Building upon the theory of |classification| , in what follows, we assume that we have a supervised dataset :math:`\mathcal{D} = \{(\mathbf{x}_1,\mathbf{y}_1),...,(\mathbf{x}_n,\mathbf{y}_n)\}` where :math:`\mathbf{x}_i\in\mathbb{R}^{m}` is a set of :math:`m` real valued measurements or features for the :math:`i`-th observation, and :math:`\mathbf{y}_i\in\{0,1\}^c` indicates which of the :math:`c` classes the :math:`i`-th observation attributes to.

.. figure:: ../../_static/how-to-ml/supervised-learning/logistic-regression/2d-example.png
    :align: center
    :figwidth: 70 %
    
    Example illustration of Logistic Regression.

Example simulated data for (binary) logistic regression. On the left, we have our dataset with a clear linear boundary separating the two classes. The right is a linear transform which separates the two classes about the origin and quantifies the probability of belonging to class 1.

In the case of binary classification i.e. only two classes, the objective of logistic regression is very clear from the above plots; We wish to find the equation of the linear boundary separating the two classes. When the data is more than 2D then we wish to find the boundary hyperplane. For binary logistic regression :math:`\mathbf{z} := \mathbf{Xw} + b` and so the gradients of the two parameters are given by :math:`\nabla_{\mathbf{w}} \mathbf{z} = \mathbf{X}` and :math:`\nabla_{b} \mathbf{z} = \mathbb{1}`. Plugging both into Eq. (14) in |binary-classification| , we yield

.. math::
    :nowrap:

    \begin{align}
        \mathbf{g}_{t + 1} &\leftarrow \mathbf{y} - \hat{\mathbf{y}}, \\
        \mathbf{w}_{t + 1} &\leftarrow \mathbf{w}_t + \alpha \mathbf{X}^\text{T}\mathbf{g}_{t+1}, \label{eq:w}\\
        b_{t+1} &\leftarrow b_t + \alpha \mathbf{1}^\text{T}\mathbf{g}_{t + 1}, \label{eq:b}
    \end{align}

where :math:`\alpha > 0` is referred to as the step-size or learning rate, and :math:`\mathbf{\hat{y}}_t := \phi(\mathbf{Xw}_t + b_t)`.

.. admonition:: Python
    :class: code

    .. raw:: html

        <details>
        <summary>Logistic Regression class</summary>
        <br>

    .. code-block:: python
        :linenos:

        from   sklearn.model_selection import train_test_split
        from   sklearn                 import datasets as ds
        import numpy as np

        class LogisticRegression():
            """
            Logistic Regression class
            
            Parameters
            =================
                loss         : function
                               Loss function to be evaluated at every time step.
                               
                phi          : function
                               Function to map the log-odds to probability space.
                               
                momentum     : float
                               Momentum for the gradient optimisation (speeds up convergence).
                               
                random_state : int
                               Parameter to be used in numpy.random.seed for reproducible results
            """
            def __init__(self, loss, phi, momentum = 0.99, random_state = None):
                self.loss         = loss
                self.phi          = phi
                self.momentum     = momentum
                self.random_state = random_state
            
            def fit(self, X, Y, X_val = None, Y_val = None, alpha = 1e-6, epochs = 2000):
                
                np.random.seed(self.random_state)

                # Detect if validation data has been given
                v = isinstance(X_val, np.ndarray)
                
                # Initialise W and b
                # Common to initialise W randomly and b at zero
                m = X.shape[1]
                c = 1 if Y.ndim == 1 else Y.shape[1]

                if c == 1:
                    W = self.W = np.random.normal(scale = 0.01, size = m)
                    b = self.b = 0
                else:
                    W = self.W = np.random.normal(scale = 0.01, size = (m, c))
                    b = self.b = np.zeros(c)

                if v:
                    # Store all of the validation "Z" values
                    Zs = []
                    
                # Loss (and validation loss if provided)
                L    = np.zeros((epochs + 1, 1 + v))
                
                # Initialise gradients for W and b to be 0
                gW   = 0
                gb   = 0
                for i in range(epochs):
                    hat  = self.phi(X @ W + b)
                    L[i] = self.loss(Y, hat)
                    if v:
                        Zs.append(X_val @ W + b)
                        L[i, 1] = self.loss(Y_val, self.phi(Zs[-1]))
                    hat  = self.phi(X @ W + b)
                    g    = (Y - hat) / len(Yb)
                    gW  *= self.momentum
                    gb  *= self.momentum
                    gW  += X.T @ g          # Eq. (2)
                    gb  += g.sum(axis = 0)  # Eq. (3)
                    W   += alpha * gW
                    b   += alpha * gb
                hat   = self.phi(X @ W + b)
                L[-1] = self.loss(Y, hat)
                if v:
                    Zs.append(X_val @ W + b)
                    L[-1, 1] = self.loss(Y_val, self.phi(Zs[-1]))
                    Zs       = np.array(Zs)
                
                # Store the loss over time
                self.logs = dict(L = L)
                
                if v:
                    # Store the validation "Z" values over time
                    self.logs['Zs'] = Zs
                    
                return self
            
            def __call__(self, X):
                return self.phi(X @ self.W + self.b)

    .. raw:: html

        </details>

.. admonition:: Python
    :class: code

    Sigmoid and binary cross entropy loss functions

    .. code-block:: python
        
        def sigmoid(z):
            """ Logistic sigmoid function """
            return 1 / (1 + np.exp(-z))

        def binary_cross_entropy(y_true, y_pred):
            """ Binary cross entropy objective function """
            offset = 1e-8 # To ensure we never compute log(0) as this is not defined!
            return (np.log(y_pred[np.where(y_true == 1)] + offset).sum(axis = -1) + np.log(1 - y_pred[np.where(y_true == 0)] + offset).sum(axis = -1)) / len(y_true)

Now lets try this code out with some data!

********************************************************
Example 1: Binary classification with load_breast_cancer
********************************************************

The `load_breast_cancer <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>`_ dataset presents a binary classification problem whereby, given some features of an observation classify it as either Malignant or Benign.

.. include:: /data/load_breast_cancer.rst

.. admonition:: Python
    :class: code
    
    Training a Logistic Regression model to the load_breast_cancer dataset

    .. code-block:: python

        data = ds.load_breast_cancer()

        ## Print description of data
        # print(data['DESCR'])

        X, y = data['data'], data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2020)

        model = LogisticRegression(binary_cross_entropy, sigmoid, momentum = 0.95, random_state = 2020).fit(X_train, y_train, X_test, y_test, alpha = 1e-6, epochs = 500)
        L     = model.logs['L']

    

Plotting L we have:

.. figure:: /_static/how-to-ml/supervised-learning/logistic-regression/load_breast_cancer.png
    :align: center
    :figwidth: 50 %
    
    Log-likelihood of logistic model over training epoch for the load_breast_cancer dataset.

which is great as both the train and test datasets have similar :math:`\mathcal{L}` values over epochs even though we are only using the training data. We can safely say we are not overfitting nor underfitting for this particular setting.

A quick animation of our model predictions for the test data shows that we achieve a final accuracy of :math:`\approx90`!

.. raw:: html
    
    <center>
    <video style="display:block; margin: 0 auto;" muted controls>
        <source src="../../_static/how-to-ml/supervised-learning/logistic-regression/breast-cancer.mp4" type="video/mp4" width="80%">     
    </video>
    <br>
    
    Playback problem? Try <a href="../../_static/how-to-ml/supervised-learning/logistic-regression/breast-cancer.gif" target="_blank">here</a>
    <br>
    <i>Vid. 1 Logistic regression predictions over training epochs.</i>
    <br>
    </center>

*************************
Multiclass Classification
*************************

So far we have only talked about two classes. If we have more than two classes and each observation can strictly belong to a single class, we will need to change our :math:`\phi` function to the softmax function. The update rules for our weights and bias parameters remain the same as stated in Eq. :math:`\eqref{eq:w}` and Eq. :math:`\eqref{eq:b}` as shown in Eq. (33) from |classification-multiclass|.

.. admonition:: Softmax and categorical cross entropy loss functions
    :class: code

    .. code-block:: python

        def softmax(z):
            """ Softmax function """
            e = np.exp(z - z.max(axis = -1, keepdims = True))
            return e / e.sum(axis = -1, keepdims = True)

        def categorical_cross_entropy(y_true, y_pred):
            """ Categorical cross entropy objective function """
            offset = 1e-8 # To ensure we never compute log(0) as this is not defined!
            return np.log(y_pred[np.where(y_true == 1)] + offset).sum() / len(y_true)


****************************************************
Example 2: Multiclass classification with load_iris
****************************************************

The `load_iris dataset <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris>`_ is an easy multiclass classification dataset whereby the task is to classify each observation into one of three classes.

.. include:: /data/load_iris.rst

Independent Sigmoid Classifiers
-------------------------------

.. admonition:: Fitting a binary Logistic Regression model to each class in the load_iris dataset
    :class: code

    .. code-block:: python

        data = ds.load_iris()
        X, y = data['data'], data['target']
        y    = np.eye(len(np.unique(y)))[y] # one-hot encode

        # # Print description
        # print(data['DESCR'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2020)

        L = []
        Z = []

        for j in range(3):
            model = LogisticRegression(binary_cross_entropy, sigmoid, random_state = 2020).fit(X_train, y_train[:,j], X_test, y_test[:,j], alpha = 1e-3, epochs = 1000)
            
            L.append(model.logs['L'])
            Z.append(model.logs['Zs'])
            
        L, Z = map(lambda A : np.moveaxis(A, 0, -1), [L, Z])

        L    = L.sum(axis = -1)

Conveniently our test set has an equal distribution of the three classes (10 observations each). Plotting the variable L yields

.. figure:: /_static/how-to-ml/supervised-learning/logistic-regression/load_iris-sigmoid.png
    :align: center
    :figwidth: 50 %
    
    Log-likelihood of sigmoid logistic model over training epoch for the load_iris dataset. 

which looks like we have converged if we observe the test set. Taking a closer look at our model predictions over time we have

.. raw:: html
    
    <center>
    <video style="display:block; margin: 0 auto;" muted controls>
        <source src="../../_static/how-to-ml/supervised-learning/logistic-regression/iris-sigmoid.mp4" type="video/mp4" width="80%">     
    </video>
    <br>
    
    Playback problem? Try <a href="../../_static/how-to-ml/supervised-learning/logistic-regression/iris-sigmoid.gif" target="_blank">here</a>
    <br>
    <i>Vid. 2 Sigmoid logistic regression latent space over training epochs.</i>
    <br>
    </center>

The above visualisation maps our 3D :math:`\mathbf{\hat{Y}}` to a 2D space by having at each corner of the triangle, the probability of belonging to that class being 1, whilst the probability of belonging to any other class strictly :math:`0`. Anywhere inbetween will map to somewhere within the triangle. Each circle represent a datapoint and the colour of that circle represents the true :math:`\mathbf{Y}`. If our estimate is 100% correct, then each circle will belong to the same coloured region.

As the animation ends, we see that there is no correlation between the red and green classes but both have the potential to be misclassified as the blue class. Two of the blue classes were misclassified as being green.

Softmax Classifier
------------------

.. admonition:: Fitting a softmax Logistic Regression model to the load_iris dataset
    :class: code

    .. code-block:: python

        data = ds.load_iris()
        X, y = data['data'], data['target']
        y    = np.eye(len(np.unique(y)))[y] # one-hot encode

        # # Print description
        # print(data['DESCR'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2020)

        model = LogisticRegression(categorical_cross_entropy, softmax, random_state = 2020).fit(X_train, y_train, X_test, y_test, alpha = 1e-3, epochs = 1000)

        L = model.logs['L']
        Z = model.logs['Zs']

.. figure:: /_static/how-to-ml/supervised-learning/logistic-regression/load_iris-softmax.png
    :align: center
    :figwidth: 50 %
    
    Log-likelihood of softmax logistic model over training epoch for the load_iris dataset. 

.. raw:: html
    
    <center>
    <video style="display:block; margin: 0 auto;" muted controls>
        <source src="../../_static/how-to-ml/supervised-learning/logistic-regression/iris-softmax.mp4" type="video/mp4" width="80%">     
    </video>
    <br>
    
    Playback problem? Try <a href="../../_static/how-to-ml/supervised-learning/logistic-regression/iris-softmax.gif" target="_blank">here</a>
    <br>
    <i>Vid. 3 Softmax logistic regression latent space over training epochs.</i>
    <br>
    </center>

We observe that the :math:`\log \mathcal{L}` values are much better for the softmax classifier than the binary sigmoid classifier. Observing the latent space projection videos, we see that generally the softmax classifier has points closer to the corners of the triangle implying it is more sure of it's predictions albeit the same mis-classification.

.. |classification| raw:: html

    <a href="../../theories/classification.html" target= "_blank">classification</a>

.. |binary-classification| raw:: html
    
    <a href="../../theories/classification.html#binary-classification" target="_blank">classification#binary-classification</a>

.. |classification-multiclass| raw:: html

    <a href="../../theories/classification.html#multiclass-classification" target="_blank">classification#multiclass</a>
