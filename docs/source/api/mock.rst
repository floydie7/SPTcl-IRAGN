.. _mock:

***********************
Mock Catalog Generation
***********************
``MockIRAGN``: A Method to Generate Cluster AGN Catalogs

Description of Use
==================
This module is used to produce mock catalogs. These have been used extensively throughout the project to understand the empirical data described in [Floyd2024]_ Chapter 5 and selected using the code described in `Infrared-Bright AGN Selection <api/selector.html>`_ as well as to test and validate the model fitting codes described in `Bayesian model fitting <api/fitting.html>`_. The primary goal with the mock catalogs is to characterize the ability to recover the model parameters of a generating model that have been embedded into the data and, equally importantly, to be able to characterize the expected uncertainty on the recovered parameters. Below is a description of the typical use of the methods in the module. For more details, all user-facing and internal methods are described in `API Documentation`_.

The module defines a single class, ``MockIRAGN``, that exposes three user-facing methods, ``make_catalog``, ``make_theta_range_suite``, and ``make_targeted_suite``. The user only needs to initialize the class and call one of the ``make_*`` methods to generate the desired mock catalog. The method ``make_catalog`` creates a single mock catalog while ``make_theta_range_suite`` and ``make_targeted_suite`` make many catalogs to form a suite based on a range of input parameters. The ``make_theta_range_suite`` method uses ranges for the model parameters θ, η, and ζ—representing the cluster amplitude, redshift power-law slope, and mass power-law slope respectively—to generate a wide range of possible catalogs. The catalogs from this method are used to create a relationship between the cluster amplitude parameter and the survey signal-to-noise ratio defined in [Floyd2024]_ §4.2. The survey signal-to-noise ratio associated with the empirical data can be used to generate mock catalogs over the range of redshift and mass power-law slopes using the ``make_targeted_suite`` method. Internally, both the suite methods call the ``make_catalog`` method to create the individual catalogs.

The process of creating a mock catalog is described here. First cluster-level information (e.g., cluster center coordinates, halo mass, redshift) are randomly sampled from the empirical dataset along with the empirical mask images generated as part of the infrared-bright selection pipeline (see `Infrared-Bright AGN Selection <api/selector.html>`_ for more details). This data is used to create a database that other methods will reference.

The internal method ``_generate_mock_catalog`` performs most of the operations in creating a single line-of-sight catalog. The following process is repeated for each galaxy cluster catalog that is to be simulated. After setting up the boundary of the "pixel" grid upon which the objects making up the line-of-sight catalog will be placed, the method first creates a background-only sub-catalog using a homogeneous spatial Poisson point process. This is designed to simulate the empirical rate of infrared-bright AGN as measured locally around the galaxy clusters (see [Floyd2024]_ §5.2 for details). The objects are then "painted on" by randomly drawing empirical galaxy-level data (e.g., photometric completeness correction, fuzzy degree of being selected as an AGN, *J*-band absolute magnitude) and associating the data with objects placed by the Poisson process.

The cluster-only sub-catalog is created by first generating a homogeneous spatial Poisson point process using the maximum of the model values with the input parameters as the input rate. The cluster candidate objects are then "painted" with galaxy-level data just as the background objects are. The model rates at each location of the placed cluster candidate objects are calculated and weighted by the maximum model rate. This ratio is used as a probability for rejection sampling. This has the effect of inducing an spatially inhomogeneous rate into the Poisson process and properly encodes any radial dependency of the generating model into the data.

The two sub-catalogs are then stacked to create the line-of-sight catalog. Cluster-level data is then "painted" onto the entire line-of-sight catalog. Projected radial distances to each object in the catalog are calculated for several centers, the true cluster center and several "miscentered" cluster centers. The miscentered centers are created by randomly shifting the measured cluster center away from the true center according to a 2D normal distribution with covariance matrix proportional to a user-specified factor of the measured telescope positional uncertainty (nominally the South Pole Telescope's 150 GHz positional uncertainty). The miscentered measurements of the radial distances allow for study of the ability to constrain the radial model parameters given uncertainty of the exact location of the galaxy cluster center.

The line-of-sight catalog is then finalized by ensuring that all objects are on "good" regions of the mask image, just as is done with the empirical data. This induces a spatial incompleteness to the data as there may be missing regions of the field of view in the empirical data. Additionally, a rejection sampling based on the photometric completeness value of the objects is performed to induce photometric incompleteness into the data.

Once all line-of-sight catalogs are generated, they are stacked into a single catalog and written out as a file. If the user is generating a suite of catalogs, the process is repeated for each set of input parameters.


.. _API Documentation:
API Documentation
=================

Cluster Information Dataclass
-----------------------------

.. autoclass:: SPTclIRAGN.Mock_Generation._ClusterInformation

Mock Catalog Builder
--------------------

.. autoclass:: SPTclIRAGN.Mock_Generation.MockIRAGN
    :members:
    :private-members:

.. [Floyd2024] Floyd, Benjamin. May 2024. “Infrared‑Bright Active Galactic Nuclei in Massive Galaxy Clusters”. Available at https://hdl.handle.net/10355/101862. Doctoral Dissertation. 5110 Rockhill Road, Kansas City, MO 64110: University of Missouri–Kansas City.