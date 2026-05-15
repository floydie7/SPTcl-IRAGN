.. _selector:

*****************************
Infrared-Bright AGN Selection
*****************************
``SelectIRAGN``: A Modular Selection Pipeline for IR-bright AGN

Description of Use
==================
This module is at the core of the project. It provides the base class ``SelectIRAGN`` and sub-classes ``SelectSDWFS`` and ``SelectFullFieldSDWFS`` which contain methods that make up the selection pipeline to generate the infrared-bright AGN catalogs in the various fields of the project. In all use cases of the selection pipelines the user only needs to initialize the selector object and run the convenience method, ``run_selection``, with any inputs specific to the particular catalog being generated. Alternatively, the user can call the individual pipeline methods to modify the specifics of the process. Below is a description of the typical process of running the selection pipeline. For a full description of all user-facing methods for the selection pipelines see `API Documentation`_.

The selection pipeline begins by generating a database of information about each cluster being processed that will be referenced throughout the pipeline. Additional information is added to the database as the pipeline progresses.

A mask image is also built by the pipeline. The mask image is an array of binary pixel values where a ``1`` indicates that the pixel is to be used later in the pipeline as well as in later analysis. The mask is first constructed by requiring pixels in both the IRAC 3.6 µm and 4.5 µm coverage images have been exposure values of at least a user-specified minimum value. The masks are then modified for clusters that have bright foreground objects or artifacts in either IRAC science images previously identified by visual inspection to remove any pixels within the regions of suspect photometry.

The IRAC photometic catalog is read in and the initial AGN selection is performed by applying the magnitude cuts described in [Floyd2024]_ §3.2 and requiring all objects to have coordinates corresponding to "good" pixels in the mask image. This preliminary catalog is then further refined by removing any remaining stars in the field of view by matching to the *Gaia* source catalog.

For each object in the catalog, in lieu of applying a standard straight color cut, fuzzy degrees of membership are calculated to determine if an object should be classified as an AGN. To recover the behavior of a standard straight color cut, the user only needs to impose a cut of :math:`\mu_\mathrm{AGN} = 0.5` to the catalog. For more details see [Floyd2024]_ §3.2. Additionally, each object has a rest-frame *J*-band absolute magnitude computed using the provided spectral energy distribution and the IRAC 3.6 µm apparent magnitude from K-corrections as defined in [Hogg2002]_. The IRAC 4.5 µm photometric completeness values for each object are added from previously computed simulations as described in [Floyd2024]_ §3.2.

Cluster-level information (e.g., the cluster mass and redshift) is then added to the catalog and projected radial distances for each object with respect to the cluster center. Finally, the individual catalogs are merged into a final single catalog and written out as a file.

Example
=======
To use the selection method we only need to initialize the selector object then run then ``run_selection`` method to generate the catalog.

>>> selector = SelectIRAGN(
        se_cat_dir=catalog_directory,
        irac_image_dir=image_directory,
        region_file_dir=regions_directory,
        mask_dir=masks_directory,
        gaia_cat_dir=gaia_catalog_directory,
        spt_catalog=cluster_catalog,
        completeness_file=completeness_sim_results,
        purity_color_threshold_file=sdwfs_purity_color_threshold,
        sed=polletta_qso2,
        irac_filter=irac_filter,
        j_band_filter=j_band_filter)
>>> agn_catalog = selector.run_selection(
        included_clusters=None,
        excluded_clusters=clusters_to_exclude,
        max_image_catalog_sep=max_separation,
        ch1_min_cov=ch1_min_coverage,
        ch2_min_cov=ch2_min_coverage,
        ch1_bright_mag=ch1_bright_mag,
        ch1_faint_mag=ch1_faint_mag,
        ch2_bright_mag=ch2_bright_mag,
        selection_band_faint_mag=ch2_faint_mag,
        spt_colnames=spt_column_names,
        output_name=None,
        output_colnames=output_column_names,
        output_coldescriptions=output_column_descriptions,
        output_colunits=output_column_units)

.. _API Documentation:
API Documentation
=================


Cluster Information Dataclass
-----------------------------

.. autoclass:: SPTclIRAGN.Pipeline_functions._ClusterInformation
    :members:

Base AGN Selection
------------------

.. autoclass:: SPTclIRAGN.Pipeline_functions.SelectIRAGN
    :members:

Postage Stamp SDWFS Selection
-----------------------------

.. autoclass:: SPTclIRAGN.Pipeline_functions.SelectSDWFS
    :members:

Full-field SDWFS Selection
--------------------------

.. autoclass:: SPTclIRAGN.Pipeline_functions.SelectFullFieldSDWFS
    :members:

.. [Floyd2024] Floyd, Benjamin. May 2024. “Infrared‑Bright Active Galactic Nuclei in Massive Galaxy Clusters”. Available at https://hdl.handle.net/10355/101862. Doctoral Dissertation. 5110 Rockhill Road, Kansas City, MO 64110: University of Missouri–Kansas City.

.. [Hogg2002] Hogg, D. W., Baldry, I. K., Blanton, M. R., & Eisenstein, D. J. 2002, arXiv e-prints, astro, doi: 10.48550/arXiv.astro-ph/0210394