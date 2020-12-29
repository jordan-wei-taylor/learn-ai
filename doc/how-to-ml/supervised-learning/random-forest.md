.. warning:: Page in development.

Random Forest
#############

The Random Forest is an ensemble method for both regression and classification tasks. It works by taking an average of a collection of *weaker* Decision trees. So first we need to understand a single Decision tree, how to make it *weaker* and then aggregate a collection of these to a Random Forest.

Decision Tree
=============

The Decision tree algorithm is a partitioning algorithm that queries an observation :math:`\mathbf{x}_i\in\mathbb{R}^m` and based on these queries will provide some estimator of the associated target value. The Decision tree can be visualised as a collection of query nodes each containing two branches and eventually ending up at a leaf node which has no branches. 

.. _decision-tree-diagram:
.. figure:: /_static/how-to-ml/supervised-learning/random-forest/decision-tree-diagram.png
    :align: center
    :figwidth: 70 %

    Example diagram of a Decision Tree. We start from the top red query node where :math:`j` indicates the feature to be queried and :math:`v` the value to be queried. If True, traverse left otherwise, traverse right until you reach a grey leaf node which represents :math:`\hat{y}`.

.. rubric:: Example use of a Decision Tree

Consider :math:`\mathbf{x} = [8, 0, 5]`. Using the above :numref:`decision-tree-diagram` we would

+ query :math:`8 \leq 7 \Rightarrow` False so we traverse right
+ query :math:`0 \leq 1 \Rightarrow` True so we traverse left
+ query :math:`5 \leq 6 \Rightarrow` True so we end up with :math:`\hat{y} = 9`

The challenge is to come up with the queries that bests partition the data such that we can provide the best estimator for the target values. Normally we act greedily based on some objective function to decide which of the :math:`m` features should be queried and what value :math:`v` should be considered for the query. For simplicity we will consider two objectives - one for regression, and one for classification. Similarly, we can define a leaf function which is used at a leaf node within our Decision Tree. The two most popular objective function for each task is variance and entropy respectively. Mathematically, they are both defined as

.. math::
    :nowrap:
    
    \begin{align}
        V(y) &= \frac{1}{n}\sum_{i=1}^n (y_i - \bar{y})^2\\\nonumber\\
        \mathcal{E}(p) &= \sum_{i=1}^n -p_i \log p_i
    \end{align}

As we will later weight these functions by the number of samples are present, we can multiply both equations by :math:`n`. Below we implement the objective and leaf functions.

.. admonition:: Python
    :class: code
    
    Decision Tree Objectives

    .. code-block:: python
    
        def variance(y):
            """ Computes the weighted variance """
            return np.square(y - y.mean()).sum()

        def entropy(y):
            """ Computes the weighted entropy """
            n = len(y)
            p = np.unique(y, return_counts = True)[1] / n
            return -p @ np.log(p) * n
    
    Leaf Functions

    .. code-block:: python
    
        def average(y):
            """ Computes the average """
            return np.mean(y, axis = 0)
        
        def proportion(c):
            """ Returns proportion leaf function """
            
            def _proportion(y):
                """ Computes the proportion of class appearance """
                return np.eye(c)[y].mean(axis = 0)
            
            # enables function to be computed in parallel if named the same
            _proportion.__qualname__ = f'proportion{c}' 
    
            return _proportion

As we will try both a regression and classifcation task lets define a Metrics class to help us.

.. admonition:: Python
    :class: code

    Metrics Class

    .. code-block:: python

        class Metrics():
            """
            Metrics class 
            
            To be inherited by algorithms for regression and classification tasks.
            """
            def rmse(self, *data, **kwargs):
                """ Root Mean-Squared Error (Regression)"""
                return np.array([np.sqrt(np.mean(np.square(self(X, **kwargs) - y))) for (X, y) in data])
            
            def acc(self, *data, **kwargs):
                """ Accuracy (Classification) """
                ret = []
                for (X, Y) in data:
                    hat = self(X, **kwargs).argmax(axis = 1)
                    y   = Y.argmax(axis = 1) if Y.ndim == 2 else Y
                    ret.append((hat == y).mean())
                return np.array(ret)
                
            def confusion_matrix(self, *data, **kwargs):
                """ Confusion Matrix (Clasification) """
                ret = []
                for (X, Y) in data:
                    hat  = self(X, **kwargs).argmax(axis = 1)
                    y    = Y.argmax(axis = 1) if Y.ndim == 2 else Y
                    u    = np.unique(y)
                    c    = len(u)
                    conf = np.empty((c, c))
                    for i in range(c):
                        for j in range(c):
                            conf[i,j] = ((y == u[i]) & (hat == u[j])).mean()
                    ret.append(conf)
                return np.array(ret)

Finally, using the code so far, we can look at implementing the Decision Tree class.

.. admonition:: Python
    :class: code
    
    .. raw:: html
    
        <details>
        <summary>Decision Tree Class</summary>

    .. code-block:: python
        :linenos:

        class DecisionTree(Metrics):
            """
            Decision Tree class
            
            Parameters
            ==============
                objective : function
                            Objective function to minimise when fitting.
                            
                leaf      : function
                            Leaf function to use when at a leaf node.
                            
                max_depth : int
                            Maximum depth level.
            """
            def __init__(self, objective, leaf, max_depth = np.inf):
                self._params   = dict(objective = objective, leaf = leaf, max_depth = max_depth)
                
            def fit(self, X, y, depth = 0):
                
                # precompute the leaf value
                leaf = self._params['leaf'](y)

                # stopping condition
                #   • max depth has been reached
                #   • there is only a single unique value
                if depth == self._params['max_depth'] or len(np.unique(y)) == 1:
                    return dict(leaf = leaf)

                # exhaustive search
                best = np.inf
                for j, x in enumerate(X.T):
                    ux = np.unique(x)
                    for v in ux[:-1]:
                        mask  = x <= v
                        score = self._params['objective'](y[mask]) + self._params['objective'](y[~mask])
                        if score < best:
                            best = score
                            tree = dict(mask = mask, v = v, j = j, leaf = leaf)

                # stopping condition
                #  • only a single unique x vector 
                if best == np.inf:
                    return dict(leaf = leaf)

                # get rid of "mask" from the tree to save memory
                mask      = tree.pop('mask')

                # recursively call on the "left" and "right" branches
                tree['l'] = self.fit(X[ mask], y[ mask], depth + 1)
                tree['r'] = self.fit(X[~mask], y[~mask], depth + 1)

                # if at the root node, save the entire tree, otherwise return this branch onwards
                if depth:
                    return tree
                self.tree = tree
                return self
            
            def _predict_one(self, x, prune = np.inf):
                """ Helper function that predicts a single x vector """
                tree  = self.tree
                depth = 0
                while 'l' in tree:
                    tree   = tree['l'] if x[tree['j']] <= tree['v'] else tree['r']
                    depth += 1
                    
                    # prune tree to evaluate as though max_depth = prune when trained
                    if depth == prune: break 
                return tree['leaf']
            
            def __call__(self, X, **kwargs):
                return np.array([self._predict_one(x, **kwargs) for x in X])

    .. raw:: html
        
        </details>

.. note::

    We can think of the *max_depth* parameter as a regularisation parameter. The lower it is, the more likely it does not overfit to the noise in the training data.

.. caution::

    This is not an exhaustive list of parameters for the Decision Tree, for a more complete list refer to sklearn's documentation!

Building a Random Forest
========================

The modern implementation of the Random Forest encorporates two elements of randomness - *bagging* [1]_ and using *random subspaces* [3]_. The original implementation of the Random Forest came from [Ho, 1995] [2]_. [Ho, 1998] [3]_ extended the original implementation of using random subspaces and [Breiman, 2001] [1]_ further extended it by introducing bagging.

Bagging
*******

The general technique of **b**-ootstrap **agg**-regat-**ing** (bagging) in context of the Random Forest involves sampling (with replacement) sets of observations to use when training each Decision Tree.

The pseudocode for our case is:

| for :math:`k` = 1,..., :math:`K`:
|   1. Sample with replacement :math:`(\mathbf{X}_k,\mathbf{y}_k)` from :math:`(\mathbf{X},\mathbf{y})`
|   2. :math:`f_k \leftarrow` DecisionTree(:math:`\mathbf{X}_k,\mathbf{y}_k`)

The Random Forest prediction function is then to take an expectation i.e. :math:`f = \mathbb{E}[f_k] = \sum_{k=1}^K f_k / K`.

Random Subspace
***************

What happens if there are a couple of dominant features? Each of our Decision Trees would query the same dominant features and would therefore be highly correlated. In attempt to remove this correlation, we can randomly select the feature space in addition to bagging the sample space.

Before we implement the Random Forest class, lets define a helper function that encorporates the random subspace sampling.

.. admonition:: Python
    :class: code

    Helper Function to train a single Decision Tree

    .. code-block:: python

        def _train_tree(X, y, features, objective, leaf, max_depth):
            """ Helper function for the Random Forest - trains a single Decision Tree with a random feature subspace"""
            return DecisionTree(objective, leaf, max_depth).fit(X[:,features], y)

Combining all we have talked about so far, we implement the Random Forest class.

.. admonition:: Python
    :class: code

    .. raw:: html

        <details>
        <summary>Random Forest Class</summary>
    
    .. code-block:: python
        :linenos:
        
        from multiprocessing import Pool # for parallel computing

        class RandomForest(Metrics):
            """
            Random Forest class
            
            Parameters
            =================
                n_estimators : int
                               Number of Decision Trees in the Random Forest.
                               
                n_features   : int, float, str
                               • int
                                   Number of features.
                               • float
                                   Proportion of total features.
                               • str
                                   ‣ "sqrt" : square-root of the total number of features.
                                   ‣ "log"  : log of the total number of features.
                                   
                objective    : function
                               Objective function to minimise when fitting.
                               
                leaf         : function
                               Leaf function to use when at a leaf node.
                               
                p_sample     : float
                               Proportion of the data each Decision Tree will see when fitting.
                               
                max_depth    : float
                               Maximum depth level.
                               
                n_processes  : int
                               Number of processes to run in parallel.
                               
                random_state : int
                               Parameter to be used in numpy.random.seed for reproducible results.
            """
            def __init__(self, n_estimators = 16, n_features = 'sqrt', objective = variance, leaf = np.mean, p_sample = 1.0, 
                         max_depth = np.inf, n_processes = 8, random_state = None):
                args         = objective, leaf, max_depth
                self._params = dict(n_estimators = n_estimators, n_features = n_features, p_sample = p_sample, 
                                    n_processes = min(n_processes, n_estimators), random_state = random_state, args = args)
                
            def fit(self, X, y):
                N, M = X.shape                            # total number of samples and features
                n    = int(self._params['p_sample'] * N)  # number of samples given to each Decision Tree
                
                # m: number of features given to each Decision Tree
                n_f  = self._params['n_features']
                if isinstance(n_f, int):
                    m = n_f
                elif isinstance(n_f, float):
                    m = int(n_f * M)
                elif n_f == 'sqrt':
                    m = int(np.sqrt(M))
                elif n_f == 'log':
                    m = int(np.log(M))
                else:
                    raise Exception()
                    
                # set random seed
                np.random.seed(self._params['random_state'])

                # subsamples of samples and features for each Decision Tree 
                rows = np.random.randint(N, size = (self._params['n_estimators'], n))
                cols = self.__features = [np.random.choice(M, replace = False, size = m) for _ in range(self._params['n_estimators'])]
                
                # train Decision Trees in parallel
                with Pool(self._params['n_processes']) as pool:
                    starmap    = ((X[r], y[r], c, *self._params['args']) for r, c in zip(rows, cols))
                    self.trees = pool.starmap(_train_tree, starmap)

                # sequential training of Decision Trees
                # self.trees = [_train_tree(X[r], y[r], c, *self._params['args']) for r, c in zip(rows, cols)]
                return self
            
            def __call__(self, X, prune = np.inf):
                # for each Decision Tree, compute the target estimates in parallel
                with Pool(self._params['n_processes']) as pool:
                    ret  = 0
                    for features, tree in zip(self.__features, self.trees):
                        starmap = ((x[features], prune) for x in X)
                        ret    += np.array(pool.starmap(tree._predict_one, starmap))
                    ret /= self._params['n_estimators'] # implies results are a simple average of each Decision Tree output
                    return ret

                # sequential computation
                # return (tree(X[:,features], prune = prune) for features, tree in zip(self.__features, self.trees)).mean(axis = 0)

    .. raw:: html

        </details>

Lets try out both the Decision Tree and Random Forest models on regression and classification problems!

.. caution::

    This is not an exhaustive list of parameters for the Random Forest, for a more complete list refer to sklearn's documentation!

    Though *max depth* is the only hyper parameter tuned below, all hyperparameters should be tuned. For conciseness and visualisation purposes, only *max depth* is tuned.


Regression
==========

For the regression problem, we try out the sklearn's load_boston dataset.

.. include:: /data/load_boston.rst

Decision Tree Regression
************************

.. admonition:: Python
    :class: code

    Loading the load_boston dataset

    .. code-block:: python

        from sklearn.datasets import load_boston

        X, y = load_boston(return_X_y = True)

        def train_val_test_split(X, y, *args, **kwargs):
            X_train, X_test, y_train, y_test = train_test_split(X      , y      , *args, **kwargs)
            X_train, X_val , y_train, y_val  = train_test_split(X_train, y_train, *args, **kwargs)
            return X_train, X_val, X_test, y_train, y_val, y_test, (X_train, y_train), (X_val, y_val), (X_test, y_test)

        X_train, X_val, X_test, y_train, y_val, y_test, train, val, test = train_val_test_split(X, y, random_state = 2020)

    RMSE results from fitting a Decision Tree varying over the maximum depth

    .. code-block:: python

        # Fit model once and evaluate at different depth levels
        depths    = range(1, 21)
        max_depth = max(depths)
        model     = DecisionTree(objective = variance, leaf = average, max_depth = max_depth).fit(X_train, y_train)
        rmse      = np.array([model.rmse(train, val, test, prune = depth) for depth in depths])
    
.. figure:: /_static/how-to-ml/supervised-learning/random-forest/dt-regression.png
    :align: center
    :figwidth: 70 %

    RMSE of Decision Tree on the load_boston dataset varying over max depth.

Random Forest Regression
************************

.. admonition:: Python
    :class: code

    .. code-block:: python
    
        model = RandomForest(objective = variance, leaf = average, max_depth = max_depth).fit(X_train, y_train)
        rmse  = np.array([model.rmse(train, val, test, prune = depth) for depth in depths])

.. figure:: /_static/how-to-ml/supervised-learning/random-forest/rf-regression.png
    :align: center
    :figwidth: 70 %

    RMSE of Random Forest on the load_boston dataset varying over max depth.

Classification
==============

For the regression problem, we try out the sklearn's load_digits dataset.

.. include:: /data/load_digits.rst

Decision Tree Classification
****************************

.. admonition:: Python
    :class: code

    Loading the load_digits dataset

    .. code-block:: python

        from sklearn.datasets import load_digits

        X, y = load_digits(return_X_y = True)

        X_train, X_val, X_test, y_train, y_val, y_test, train, val, test = train_val_test_split(X, y, random_state = 2020)

    Accuracy results from fitting a Decision Tree varying over the maximum depth

    .. code-block:: python

        # leaf function
        proportion10 = proportion(10)

        # Fit model once and evaluate at different depth levels
        model        = DecisionTree(objective = entropy, leaf = proportion10, max_depth = max_depth).fit(X_train, y_train)
        acc          = np.array([model.acc(train, val, test, prune = depth) for depth in depths])
    
.. figure:: /_static/how-to-ml/supervised-learning/random-forest/dt-classification.png
    :align: center
    :figwidth: 70 %

    Accuracy of Decision Tree on the load_digits dataset varying over max depth.

Random Forest Classification
****************************

.. admonition:: Python
    :class: code

    Accuracy results from fitting a Random Forest varying over the maximum depth

    .. code-block:: python
    
        model = RandomForest(objective = entropy, leaf = proportion10, max_depth = max_depth).fit(X_train, y_train)
        acc   = np.array([model.acc(train, val, test, prune = depth) for depth in depths])

.. figure:: /_static/how-to-ml/supervised-learning/random-forest/rf-classification.png
    :align: center
    :figwidth: 70 %

    Accuracy of Random Forest on the load_digits dataset varying over max depth.

.. rubric:: References

.. [1] L. Breiman, *Random Forests*, 2001, https://link.springer.com/article/10.1023/A:1010933404324
.. [2] T. K. Ho, *Random Decision Forests*, 1995, https://ieeexplore.ieee.org/document/598994
.. [3] T. K. Ho, *The Random Subspace Method for Constructing Decision Forests*, 1998, https://ieeexplore.ieee.org/document/709601
