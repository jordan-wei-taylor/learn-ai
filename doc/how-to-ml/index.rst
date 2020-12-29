.. warning:: Page in development.

How to: Machine Learning
########################

Tutorial series for those who are new into the machine learning community and want to learn from the ground up. Generally, machine learning can be split up into two main branches; supervised, and unsupervised learning. The former has the assumption that our dataset has both the observations, commonly denoted by :math:`\mathbf{X}` and the associated targets :math:`\mathbf{Y}`. It is then of interest to learn some functional :math:`f`, which requires some parameters :math:`\boldsymbol{\theta}`, that explains how each and every :math:`\mathbf{x}_i` maps to an estimate of :math:`\mathbf{y}_i`. Then, by quantifying some error metric, :math:`\mathcal{L}(\boldsymbol{\theta};\mathbf{X},\mathbf{Y})`, we optimise :math:`\boldsymbol{\theta}` that minimises the quantity :math:`\mathcal{L}(\boldsymbol{\theta};\mathbf{X},\mathbf{Y})`. Alternatively, we may not have any targets and want to learn the topology or structure of our data :math:`\mathbf{X}`. This is generally the harder and the optimisation steps are very task specific.

The *How to: Machine Learning* tutorial series aims to go through the foundations of machine learning and the popular algorithms it is comprised of. For the applications of machine learning, see the `Advanced Applications <../advanced-applications/index.html>`_ tutorial series which assumes you have the foundations under your belt.

.. rubric:: Table of Contents

.. toctree::
   :maxdepth: 2
    
   Supervised Learning <supervised-learning/index.rst>
   Unsupervised Learning <unsupervised-learning/index.rst>
