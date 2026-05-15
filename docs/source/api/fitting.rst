.. _fitting:

*************************
Bayesian Modeling Fitting
*************************

Description of Use
==================
The primary data analysis for this project is carried out by two modules, a preprocessing script and the sampler script. These scripts have been optimized to work on a high-performance computing cluster using Message Passing Interface (MPI) for parallelization. They are designed to work in sequence, first the preprocessing script runs, then the sampler script takes in the output of the preprocessing script. Together they perform the Bayesian statistical model fitting that makes up the primary analysis discussed in [Floyd2024]_ §4.1. Below is a description of how the two scripts work and interact. For more details on individual components within the scripts, see `API Documentation`_.

The preprocessing script has two primary tasks: (a) to take in all the input data such as the mask images and infrared-bright AGN catalog (see :ref:`Infrared-Bright AGN Selection _selector` for details on how they are created) and generate a database of only relevant data needed in the sampler script and (b) for each mask image, compute the fraction of "good" pixels in concentric annuli that will be used as weights to an integration performed in the sampler. The creation of the intermediate database file allows for predictable input for the sampler script and a reduction of required memory needed while the sampler is being ran. Additionally, for any subsequent runs of the sampler the preprocessing only needs to be ran again if the input data changes. This allows the user to easily and efficiently run multiple tests on different variations of the sampler on the same input data.

The sampler script implements the Bayesian model fitting analysis. It takes in only the preprocessing database file as input which contains all information needed by the sampler functions. This script uses the ``emcee`` library developed by [ForemanMackey2013]_ for the Markov chain Monte Carlo (MCMC) sampler that produces the posterior probability distribution. The sampler script defines several functions that are used by the ensemble sampler class from ``emcee``. The ``lnpost`` function is the unmarginalized log-posterior probability on which the MCMC sampler will be applied. It passes the proposed model parameter values to the log-likelihood and log-prior probability functions. The ``lnprior`` function defines the prior probability functions and will return a value proportional to the sum of the individual prior probability functions evaluated at the parameter values or a value of -∞ if the proposed parameter value is outside the defined bounds. The ``lnlike`` function is the log-likelihood function. It is defined as a Poisson probability distribution that has been modified to reflect the incomplete nature of the input data. This can be seen in Equation (4.4) in [Floyd2024]_. This function calls ``model_rate_opted`` which defines the Poisson intensity function as defined in Equation (4.1) of [Floyd2024]_. It also uses a modified trapezoidal integration where each of the sub-intervals are weighted according to the fraction of "good" pixels as calculated in the preprocessing script beforehand.

The sampler script will check the MCMC chain for convergence every 100 steps. Convergence is satisfied when the mean autocorrelation time from the previous and the current iterations have a relative error of no larger than 1%. Once convergence is reached the sampler will end. Otherwise, the sampler will continue until a maximum iteration count is reached. The MCMC chain and related data created by the ``emcee`` sampler will be periodically saved to a file as the script is being executed. This allows for checkpointing and resuming of the MCMC sampler if needed. Once the script is finished, the results file can then be analyzed to determine the final, marginalized posterior probability distributions of the model parameters.

.. _API Documentation:
API Documentation
=================

Preprocessing Functions
-----------------------

.. automodule:: SPTclIRAGN.SPT_AGN_emcee_preprocessing
    :members:

MCMC Sampler Functions
----------------------

.. automodule:: SPTclIRAGN.SPT_AGN_emcee_sampler_MPI
    :members:

.. [Floyd2024] Floyd, Benjamin. May 2024. “Infrared‑Bright Active Galactic Nuclei in Massive Galaxy Clusters”. Available at https://hdl.handle.net/10355/101862. Doctoral Dissertation. 5110 Rockhill Road, Kansas City, MO 64110: University of Missouri–Kansas City.
.. [ForemanMackey2013] Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. 2013, PASP, 125, 306, doi: 10.1086/670067