"""
Mock_Generation.py
Author: Benjamin Floyd

Provides a class-oriented approach to generating the mock catalogs.
"""
import glob
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, QTable, join, unique, Column, vstack
from astropy.units.format import fits
from astropy.wcs import WCS
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from scipy.interpolate import lagrange, interp1d
from scipy.stats import stats
from synphot import SpectralElement, SourceSpectrum, units

from SPTclIRAGN.utils.k_correction import k_corr_abs_mag


@dataclass(slots=True, kw_only=True, order=True)
class _ClusterInformation:
    """Dataclass containing information about the cluster and line of sight properties."""
    spt_id: str
    mask_name: str
    z_cl: float
    m500_cl: u.Quantity
    r500_cl: u.Quantity
    sz_center: SkyCoord
    sz_theta_core: u.Quantity
    sz_xi: float
    color_threshold: float
    mask_wcs: WCS = field(init=False)
    mask_pixel_scale: u.Quantity = field(init=False)
    mask_size_x: int = field(init=False)
    mask_size_y: int = field(init=False)
    mask_upper_x: int = None
    mask_upper_y: int = None
    mask_lower_x: int = None
    mask_lower_y: int = None
    r_grid: NDArray[float] = None
    j_grid: NDArray[float] = None

    def __post_init__(self):
        # Populate the wcs related fields automatically
        self.mask_wcs = WCS(self.mask_name)
        self.mask_pixel_scale = self.mask_wcs.proj_plane_pixel_scales()[0]

        # For the mask's image size we need to subtract 1 to account for the shift between index and length
        self.mask_size_x = self.mask_wcs.pixel_shape[0] - 1
        self.mask_size_y = self.mask_wcs.pixel_shape[1] - 1


class MockIRAGN:
    def __init__(self,
                 cosmo: FlatLambdaCDM,
                 rng_seed: int,
                 lf_z: list[float],
                 lf_m_star: list[float],
                 lf_phi_star: list[float],
                 lf_36_filter: SpectralElement,
                 lf_45_filter: SpectralElement,
                 lf_j_filter: SpectralElement,
                 lf_sed: SourceSpectrum,
                 redshift_bins: list[float],
                 color_thresholds: list[float],
                 galaxy_data_catalog: Table,
                 cluster_data_catalog: Table,
                 local_bkg_colors: Column,
                 local_bkg_mean: Column,
                 local_bkg_std: Column):
        r"""
        Using our Bayesian generating model, create a mock catalog for use in testing limitations of both the model 
        itself and the fitting procedures.
        
        Parameters
        ----------
        cosmo
            Cosmology object.
        rng_seed
            An integer used to define the RNG used in all random processes.
        lf_z
            Redshift grid used in defining the luminosity function parameters.
        lf_m_star
            Characteristic absolute magnitude grid defining the luminosity function parameter :math:`M^{*}(z_{i})`.
        lf_phi_star
            Normalization factor grid defining the luminosity function parameter :math:`\log\phi^{*}(z_{i})`.
        lf_36_filter
            IRAC 3.6 um filter used in performing K-corrections on observed photometry into J-band absolute magnitudes.
        lf_45_filter
            IRAC 4.5 um filter used in performing K-corrections on observed photometry into J-band absolute magnitudes.
        lf_j_filter
            J-band filter used in performing K-corrections on observed photometry into J-band absolute magnitudes.
        lf_sed
            Spectral Energy Distribution of assumed type of galaxy to perform K-corrections on observed photometry into
            J-band absolute magnitudes.
        redshift_bins
            Redshift bins used in several interpolations related to the AGN threshold colors and surface densities.
        color_thresholds
            Colors defining AGN selection thresholds associated with redshift bins.
        galaxy_data_catalog
            Catalog of ancillary data to be painted onto simulated objects on a per-galaxy basis.
        cluster_data_catalog
            Catalog of ancillary data to be painted onto simulated objects on a per-cluster basis.
        local_bkg_colors
            Column of color bins associated with the local background AGN surface density parameters.
        local_bkg_mean
            Column of local background AGN surface density means.
        local_bkg_std
            Column of local background AGN surface density standard deviations.
        """

        self._cosmo = cosmo
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(rng_seed)

        # Set luminosity function parameters
        self._m_star = lagrange(lf_z[1:], lf_m_star)
        self._log_phi_star = lagrange(lf_z, lf_phi_star)

        # Set K-correction parameters
        self._irac_36_filter = lf_36_filter
        self._irac_45_filter = lf_45_filter
        self._flamingos_j_filter = lf_j_filter
        self._qso2_sed = lf_sed

        # Build color--redshift and surface density--color interpolation
        self._agn_purity_color = interp1d(redshift_bins, color_thresholds)

        # Build interpolators for local background AGN surface density parameters
        self._local_bkg_mean = interp1d(local_bkg_colors, local_bkg_mean)
        self._local_bkg_std = interp1d(local_bkg_colors, local_bkg_std)

        # Set ancillary catalogs
        self._galaxy_data_cat = galaxy_data_catalog
        self._cluster_data_cat = cluster_data_catalog

        # Create a RegEx pattern for the cluster ID
        self._cluster_id_pattern = re.compile(r'SPT-CLJ\d+-\d+')

        # Collection for line-of-sight data
        self._los_data: set | None = None

        # Used in final catalog names
        self._c0_true = 0.181

    def make_targeted_suite(self,
                            num_clusters: int,
                            mask_files: list[str],
                            snr_target: float,
                            snr_theta: dict[str, Polynomial],
                            eta_range: list[float],
                            zeta_range: list[float],
                            beta_true: float,
                            rc_true: float,
                            max_radius: float,
                            irac_45_bright_mag: float,
                            irac_45_faint_mag: float,
                            cluster_delta_z: float,
                            sz_theta_beam: u.Quantity,
                            miscentering_offsets: list[float],
                            output_dir: Path):
        r"""
        Creates a suite of mock catalogs over a range of redshift and mass parameter values. Uses a targeted SNR to 
        determine the cluster amplitude parameter. 
        
        Parameters
        ----------
        num_clusters
            Number of clusters to generate in each catalog.
        mask_files
            List of empirical masks to draw mock clusters from.
        snr_target
            Specified signal-to-noise ratio that determines the value of the cluster amplitude parameter, 
            :math:`\\theta` given redshift and mass parameters, :math:`\eta` and :math:`\zeta` respectively.
        snr_theta
            A dictionary containing the fits of SNR--theta for various eta and zeta parameters.
        eta_range
            Range of redshift parameter values to generate each catalog in the suite.
        zeta_range
            Range of mass parameter values to generate each catalog in the suite.
        beta_true
            Radial beta-model slope parameter value.
        rc_true
            Radial beta-model scale parameter value.
        max_radius
            Maximum radius in units of r500 to generate objects along each field of view.
        irac_45_bright_mag
            Bright-end cut in 4.5 um used in empirical AGN selection.
        irac_45_faint_mag
            Faint-end cut in 4.5 um used in empirical AGN selection.
        cluster_delta_z
            Redshift range used to select for objects in `galaxy_data_catalog` for cluster objects. Satisfies
            :math:`|z - z_{\rm cl} | \leq \delta z`.
        sz_theta_beam
            SPT 150 GHz beam size. Used for miscentering analyses.
        miscentering_offsets
            Factors of cluster positional uncertainty on which to perform miscentering analyses.
        output_dir
            Directory to place the output catalog files in.
        """

        # Iterate through all possible combinations of the redshift and mass parameters
        for eta_true, zeta_true in np.array(np.meshgrid(eta_range, zeta_range)).T.reshape(-1, 2):
            # Using the targeted SNR, determine the true cluster amplitude parameter
            theta_true = snr_theta[f'{(eta_true, zeta_true)}'](snr_target)

            # Generate the line-of-sight catalog
            self.make_catalog(num_clusters, mask_files, theta_true, eta_true, zeta_true, beta_true, rc_true,
                              max_radius, irac_45_bright_mag, irac_45_faint_mag, cluster_delta_z, sz_theta_beam,
                              miscentering_offsets, output_dir)

    def make_theta_range_suite(self,
                               num_clusters: int,
                               mask_files: list[str],
                               theta_range: list[float],
                               eta_range: list[float],
                               zeta_range: list[float],
                               beta_true: float,
                               rc_true: float,
                               max_radius: float,
                               irac_45_bright_mag: float,
                               irac_45_faint_mag: float,
                               cluster_delta_z: float,
                               sz_theta_beam: u.Quantity,
                               miscentering_offsets: list[float],
                               output_dir: Path):
        r"""
        Creates a suite of mock catalogs over a range of cluster amplitude, redshift, and mass parameter values.

        Parameters
        ----------
        num_clusters
            Number of clusters to generate in each catalog.
        mask_files
            List of empirical masks to draw mock clusters from.
        theta_range
            Range of cluster amplitude values to generate each catalog in the suite
        eta_range
            Range of redshift parameter values to generate each catalog in the suite.
        zeta_range
            Range of mass parameter values to generate each catalog in the suite.
        beta_true
            Radial beta-model slope parameter value.
        rc_true
            Radial beta-model scale parameter value.
        max_radius
            Maximum radius in units of r500 to generate objects along each field of view.
        irac_45_bright_mag
            Bright-end cut in 4.5 um used in empirical AGN selection.
        irac_45_faint_mag
            Faint-end cut in 4.5 um used in empirical AGN selection.
        cluster_delta_z
            Redshift range used to select for objects in `galaxy_data_catalog` for cluster objects. Satisfies
            :math:`|z - z_{\rm cl} | \leq \delta z`.
        sz_theta_beam
            SPT 150 GHz beam size. Used for miscentering analyses.
        miscentering_offsets
            Factors of cluster positional uncertainty on which to perform miscentering analyses.
        output_dir
            Directory to place the output catalog files in.
        """

        # Iterate through all possible combinations of the redshift and mass parameters
        for theta_true, eta_true, zeta_true in np.array(np.meshgrid(
                theta_range, eta_range, zeta_range)).T.reshape(-1, 3):
            # Generate the line-of-sight catalog
            self.make_catalog(num_clusters, mask_files, theta_true, eta_true, zeta_true, beta_true, rc_true,
                              max_radius, irac_45_bright_mag, irac_45_faint_mag, cluster_delta_z, sz_theta_beam,
                              miscentering_offsets, output_dir)

    def make_catalog(self,
                     num_clusters: int,
                     mask_files: list[str],
                     theta_true: float,
                     eta_true: float,
                     zeta_true: float,
                     beta_true: float,
                     rc_true: float,
                     max_radius: float,
                     irac_45_bright_mag: float,
                     irac_45_faint_mag: float,
                     cluster_delta_z: float,
                     sz_theta_beam: u.Quantity,
                     miscentering_offsets: list[float],
                     output_dir: Path):
        r"""
        Creates a catalog of cluster line-of-sights with the specified input parameters.
        
        Parameters
        ----------
        miscentering_offsets
        num_clusters
            Number of clusters to generate in each catalog.
        mask_files
            List of empirical masks to draw mock clusters from.
        theta_true
            Cluster amplitude parameter.
        eta_true
            Redshift power-law slope parameter.
        zeta_true
            Mass power-law slope parameter.
        beta_true
            Radial beta-model slope parameter value.
        rc_true
            Radial beta-model scale parameter value.
        max_radius
            Maximum radius in units of r500 to generate objects along each field of view.
        irac_45_bright_mag
            Bright-end cut in 4.5 um used in empirical AGN selection.
        irac_45_faint_mag
            Faint-end cut in 4.5 um used in empirical AGN selection.
        cluster_delta_z
            Redshift range used to select for objects in `galaxy_data_catalog` for cluster objects. Satisfies
            :math:`|z - z_{\rm cl}| \leq \delta z`.
        sz_theta_beam
            SPT 150 GHz beam size. Used for miscentering analyses.
        miscentering_offsets
            Factors of cluster positional uncertainty on which to perform miscentering analyses.
        output_dir
            Directory to place the output catalog files in.
        """

        # Using our masks, create our input set of cluster data to use.
        # First, make sure that we only select clusters that have masks existing.
        mask_files = [f for f in mask_files if self._cluster_id_pattern.search(f).group(0)
                      in self._cluster_data_cat['SPT_ID']]

        # Select a number of masks at random, sorted to match the order in `SPTcl`.
        masks_bank = sorted([mask_files[i] for i in self._rng.choice(num_clusters, size=num_clusters, replace=False)],
                            key=lambda x: self._cluster_id_pattern.search(x).group(0))

        # Find the corresponding cluster IDs in the SPT catalog that match the masks we chose
        spt_catalog_ids = [self._cluster_id_pattern.search(mask_name).group(0) for mask_name in masks_bank]
        spt_catalog_mask = [self._cluster_data_cat['SPT_ID'] == spt_id for spt_id in spt_catalog_ids]
        spt_data = self._galaxy_data_cat[
            'SPT_ID', 'RA', 'DEC',
            'M500', 'R500', 'REDSHIFT',
            'THETA_CORE', 'XI', 'field'][spt_catalog_mask]

        # Create cluster names
        name_bank = [f'SPT_Mock_{i:03d}' for i in range(num_clusters)]

        # Combine our data into a catalog
        spt_data.rename_columns(['SPT_ID', 'RA', 'DEC'], ['orig_SPT_ID', 'SZ_RA', 'SZ_DEC'])
        spt_data['SPT_ID'] = name_bank
        spt_data['MASK_NAME'] = masks_bank

        # Populate the database of input data
        self._los_data = {_ClusterInformation(spt_id=cluster['SPT_ID'],
                                              mask_name=cluster['MASK_NAME'],
                                              z_cl=cluster['REDSHIFT'],
                                              m500_cl=cluster['M500'],
                                              r500_cl=cluster['R500'],
                                              sz_center=SkyCoord(cluster['SZ_RA'],
                                                                 cluster['SZ_DEC'], unit=u.deg),
                                              sz_theta_core=cluster['THETA_CORE'],
                                              sz_xi=cluster['XI'],
                                              color_threshold=self._agn_purity_color(cluster['REDSHIFT']))
                          for cluster in spt_data}

        # Generate the line-of-sight catalogs
        los_agn_catalogs = [self._generate_mock_cluster(cluster_data, theta_true, eta_true, zeta_true, beta_true,
                                                        rc_true, max_radius, irac_45_faint_mag, irac_45_bright_mag,
                                                        cluster_delta_z, sz_theta_beam, miscentering_offsets)
                            for cluster_data in spt_data]

        # Stack the individual line-of-sight catalogs into a single survey-level catalog
        survey_catalog = vstack(los_agn_catalogs)

        # Write the catalog to disk
        output_file_name = output_dir / (f'mock_AGN_catalog_t{theta_true:.4f}_e{eta_true:.2f}_z{zeta_true:.2f}'
                                         f'_b{beta_true:.2f}_rc{rc_true:.3f}_C{self._c0_true:.3f}'
                                         f'_maxr{max_radius:.2f}_seed{self._rng_seed}_{num_clusters}_tez_grid.fits')
        survey_catalog.write(output_file_name, overwrite=True)

        # Print out the mock catalog statistics
        logging.info(f'RNG Seed: {self._rng_seed}')
        logging.info('Mock Catalog (Photometrically Complete)')
        self._print_catalog_stats((theta_true, eta_true, zeta_true, beta_true, rc_true, self._c0_true),
                                  survey_catalog, phot_incomplete=False)
        logging.info('Mock Catalog (Photometrically Incomplete)')
        self._print_catalog_stats((theta_true, eta_true, zeta_true, beta_true, rc_true, self._c0_true),
                                  survey_catalog, phot_incomplete=True)

    def _generate_mock_cluster(self,
                               cluster_data: _ClusterInformation,
                               theta_true: float,
                               eta_true: float,
                               zeta_true: float,
                               beta_true: float,
                               rc_true: float,
                               max_radius: float,
                               irac_45_bright_mag: float,
                               irac_45_faint_mag: float,
                               cluster_delta_z: float,
                               sz_theta_beam: u.Quantity,
                               miscentering_offsets: list[float]) -> Table:
        r"""
        Generates a single line of sight catalog using the input parameters.

        Parameters
        ----------
        cluster_data
            Dataclass containing information about the cluster and line of sight properties.
        theta_true
            Cluster amplitude parameter.
        eta_true
            Redshift power-law slope parameter.
        zeta_true
            Mass power-law slope parameter.
        beta_true
            Radial beta-model slope parameter value.
        rc_true
            Radial beta-model scale parameter value.
        max_radius
            Maximum radius in units of r500 to generate objects along each field of view.
        irac_45_bright_mag
            Bright-end cut in 4.5 um used in empirical AGN selection.
        irac_45_faint_mag
            Faint-end cut in 4.5 um used in empirical AGN selection.
        cluster_delta_z
            Redshift range used to select for objects in `galaxy_data_catalog` for cluster objects. Satisfies
            :math:`|z - z_{\rm cl}| \leq \delta z`.
        sz_theta_beam
            SPT 150 GHz beam size. Used for miscentering analyses.
        miscentering_offsets
            Factors of cluster positional uncertainty on which to perform miscentering analyses.

        Returns
        -------
        Table
            The line-of-sight catalog.
        """

        # Find the mask's bounding box and store the data in the cluster information data structure
        self._mask_bounding_box(cluster_data, max_radius)

        # Set up grid of radial positions to place AGN on (normalized by r500)
        cluster_data.r_grid = np.linspace(0., max_radius, num=200)

        # Generate a grid of J-band absolute magnitudes on which we will compute model rates to determine the maximum
        # rate. They will be derived from the empirical 4.5 um magnitude limits k-corrected into J-band absolute
        # magnitudes.
        faint_end_j_absmag = k_corr_abs_mag(irac_45_faint_mag, z=cluster_data.z_cl, f_lambda_sed=self._qso2_sed,
                                            zero_pt_obs_band=179.7 * u.Jy, zero_pt_em_band='vega',
                                            obs_filter=self._irac_45_filter, em_filter=self._flamingos_j_filter,
                                            cosmo=self._cosmo)
        bright_end_j_absmag = k_corr_abs_mag(irac_45_bright_mag, z=cluster_data.z_cl, f_lambda_sed=self._qso2_sed,
                                             zero_pt_obs_band=179.7 * u.Jy, zero_pt_em_band='vega',
                                             obs_filter=self._irac_45_filter, em_filter=self._flamingos_j_filter,
                                             cosmo=self._cosmo)
        cluster_data.j_grid = np.linspace(bright_end_j_absmag, faint_end_j_absmag, num=200)

        # Create the background catalog
        # Generate the local background rate using the empirical local background mean and standard deviation.
        c_mean = self._local_bkg_mean(cluster_data.color_threshold)
        c_std = self._local_bkg_std(cluster_data.color_threshold)
        c_true_local = self._rng.normal(loc=c_mean, scale=c_std)
        c_true_local *= u.arcmin ** -2 * cluster_data.mask_pixel_scale.to(u.arcmin) ** 2
        bkg_cat = self._generate_sub_catalog(input_params=c_true_local, cluster_data=cluster_data, cluster=False)

        # Create the cluster catalog
        # Set our cluster model parameters
        cluster_params_true = (theta_true, eta_true, zeta_true, beta_true, rc_true)
        cl_cat = self._generate_sub_catalog(input_params=cluster_params_true, cluster_data=cluster_data, cluster=True,
                                            cluster_delta_z=cluster_delta_z)

        # Stack the sub-catalogs together to create the final line-of-sight catalog
        los_cat = vstack([cl_cat, bkg_cat])

        # Paint the cluster-level data onto the catalog
        los_cat['SPT_ID'] = cluster_data.spt_id
        los_cat['SZ_RA'] = cluster_data.sz_center.ra
        los_cat['SZ_DEC'] = cluster_data.sz_center.dec
        los_cat['REDSHIFT'] = cluster_data.z_cl
        los_cat['M500'] = cluster_data.m500_cl
        los_cat['R500'] = cluster_data.r500_cl
        los_cat['MASK_NAME'] = cluster_data.mask_name

        # Additionally, add the input mean, standard deviation, and sampled local background surface densities
        los_cat['c_mean'] = c_mean
        los_cat['c_std'] = c_std
        los_cat['c_true'] = c_true_local

        # Calculate RA/Dec coordinates for the pixel coordinates
        los_skycoords = SkyCoord.from_pixel(los_cat['x_pixel'], los_cat['y_pixel'], wcs=cluster_data.mask_wcs)
        los_cat['RA'] = los_skycoords.ra
        los_cat['DEC'] = los_skycoords.dec

        # Calculate the true projected radial distances for the final catalog in both angular and r500 units
        los_cat['RADIAL_SEP_ARCMIN'] = cluster_data.sz_center.separation(los_skycoords).to(u.arcmin)
        los_cat['RADIAL_SEP_R500'] = (los_cat['RADIAL_SEP_ARCMIN']
                                      * self._cosmo.kpc_proper_per_arcmin(los_cat['REDSHIFT']).to(u.Mpc / u.arcmin)
                                      / los_cat['R500'])

        # Additionally, compute miscentered radial distances
        for sigma_offset in miscentering_offsets:
            offset_center_coord, offset_radial_dist_arcmin, offset_radial_dist_r500 = self._miscentering(
                sigma_offset=sigma_offset, true_center=cluster_data.sz_center, object_coords=los_skycoords,
                theta_beam=sz_theta_beam, theta_core=cluster_data.sz_theta_core, xi=cluster_data.sz_xi,
                redshift=cluster_data.z_cl, r500=cluster_data.r500_cl)

            # Add the offset data to the catalog
            los_cat[f'OFFSET_{sigma_offset:.2f}_RA'] = offset_center_coord.ra
            los_cat[f'OFFSET_{sigma_offset:.2f}_DEC'] = offset_center_coord.dec
            los_cat[f'OFFSET_{sigma_offset:.2f}_RADIAL_SEP_ARCMIN'] = offset_radial_dist_arcmin
            los_cat[f'OFFSET_{sigma_offset:.2f}_RADIAL_SEP_R500'] = offset_radial_dist_r500

        # Clean up the catalog
        # Read in the mask image
        mask_image = fits.getdata(cluster_data.mask_name).astype(bool)

        # Pass the line of sight catalog through the mask. This both ensures that all objects are within the image
        # bounds and induces a spatial incompleteness to the mock catalog.
        los_cat = los_cat[
            mask_image[np.floor(los_cat['y_pixel']).astype(int), np.floor(los_cat['x_pixel']).astype(int)]]

        # Perform a rejection sampling based on the photometric completeness value for each object and store the result
        # as a flag in the catalog to allow for tests to choose whether to use the incomplete version of the mock.
        # The rejection probability will be defined as the completeness simulation recovery value  1 / correction.
        rejection_prob = 1 / los_cat['COMPLETENESS_CORRECTION']
        alpha = self._rng.uniform(0., 1., size=len(los_cat))
        los_cat['COMPLETENESS_REJECT'] = rejection_prob >= alpha

        return los_cat

    def _mask_bounding_box(self,
                           cluster_data: _ClusterInformation,
                           max_radius: float):
        """
        Sets the bounding box for the eligible region for object placement by the Poisson point process.

        Parameters
        ----------
        cluster_data
            Dataclass containing information about the cluster and line of sight properties.
        max_radius
            Maximum radius in units of r500 to generate objects along each field of view.
        """

        # Compute the maximum radius in pixel units
        max_radius_pix = (max_radius * cluster_data.r500_cl * self._cosmo.arcsec_per_kpc_proper(cluster_data.z_cl)
                          .to(cluster_data.mask_pixel_scale.unit / u.Mpc) / cluster_data.mask_pixel_scale).value

        # Set the bounding box values for object placement relative from the cluster center
        sz_center_pix = cluster_data.sz_center.to_pixel(wcs=cluster_data.mask_wcs)
        cluster_data.mask_upper_x = sz_center_pix[0] + max_radius_pix
        cluster_data.mask_upper_y = sz_center_pix[1] + max_radius_pix
        cluster_data.mask_lower_x = sz_center_pix[0] - max_radius_pix
        cluster_data.mask_lower_y = sz_center_pix[1] - max_radius_pix

    def _generate_sub_catalog(self,
                              input_params: float | tuple[float, ...],
                              cluster_data: _ClusterInformation,
                              cluster: bool,
                              cluster_delta_z: float = None) -> Table:
        """
        Generates the background and cluster sub-catalogs.

        Parameters
        ----------
        input_params
            Generating model parameter(s) to be used in the Poisson point process rate.
        cluster_data
            Dataclass containing information about the cluster and line of sight properties.
        cluster
            Flag designating if the sub-catalog is a cluster sample or a background sample.

        Returns
        -------
        Table
            The sub-catalog containing object positions and galaxy-level ancillary data painted.
        """

        # To avoid any UnboundLocalErrors we will declare the max_rate variable with a dummy value
        max_rate = 1.

        # Make a cut in the galaxy data catalog to only include objects with selection memberships >= 50%
        galaxy_data_catalog = self._galaxy_data_cat[
            self._galaxy_data_cat[f'SELECTION_MEMBERSHIP_{cluster_data.color_threshold:.2f}'] >= 0.5]

        if not cluster:
            # Generate background sub-catalog objects
            agn_pix_coords = self._poisson_point_process(rate=input_params,
                                                         dx=cluster_data.mask_upper_x,
                                                         dy=cluster_data.mask_upper_y,
                                                         lower_dx=cluster_data.mask_lower_x,
                                                         lower_dy=cluster_data.mask_lower_y)

            # For background objects, we only need to use the catalog cut at our specified selection membership.
            galaxy_data_catalog = galaxy_data_catalog

        else:
            # Generate cluster sub-catalog objects
            # Calculate the initial model values over the full range of radial and absolute magnitude values
            model_cluster_agn = self._model_rate(params=input_params,
                                                 z=cluster_data.z_cl,
                                                 m500=cluster_data.m500_cl,
                                                 r500=cluster_data.r500_cl,
                                                 r_r500=cluster_data.r_grid,
                                                 j_mag=cluster_data.j_grid)

            # Find the maximum rate. This will set the initial homogeneous spatial Poisson rate.
            # It also establishes that the number of AGN in the cluster is tied to the redshift and mass of the cluster.
            # Then convert to pix^-2 units.
            max_rate = np.max(model_cluster_agn)
            max_rate *= (cluster_data.r500_cl ** -2 * self._cosmo.kpc_proper_per_arcmin(cluster_data.z_cl)
                         .to(u.Mpc / u.arcmin) ** 2 * cluster_data.mask_pixel_scale.to(u.arcmin) ** 2)

            # Generate the initial cluster sub-catalog objects
            agn_pix_coords = self._poisson_point_process(rate=max_rate,
                                                         dx=cluster_data.mask_upper_x,
                                                         dy=cluster_data.mask_upper_y,
                                                         lower_dx=cluster_data.mask_lower_x,
                                                         lower_dy=cluster_data.mask_lower_y)

            # For cluster objects, we want to only select galaxy data that is roughly similar to the cluster redshift
            galaxy_data_catalog = galaxy_data_catalog[
                np.abs(galaxy_data_catalog['REDSHIFT'] - cluster_data.z_cl) <= cluster_delta_z]

        # Before continuing, paint the galaxies with the associated galaxy data sampled with replacement
        sub_catalog = Table.from_pandas(galaxy_data_catalog.to_pandas().sample(n=agn_pix_coords.shape[-1],
                                                                               replace=True, random_state=self._rng))

        # Add the raw cartesian coordinates to the sub-catalog
        sub_catalog['x_pixel'] = agn_pix_coords[0]
        sub_catalog['y_pixel'] = agn_pix_coords[1]

        # Rename our columns for compliance with empirical data catalogs
        sub_catalog.rename_columns(['REDSHIFT', f'SELECTION_MEMBERSHIP_{cluster_data.color_threshold:.2f}'],
                                   ['galaxy_redshift', 'SELECTION_MEMBERSHIP'])

        # Apply a flag to the objects
        sub_catalog['CLUSTER_AGN'] = np.full_like(sub_catalog['x_pixel'], fill_value=cluster)

        # Select for only the relevant columns
        sub_catalog = sub_catalog[
            'x_pixel', 'y_pixel', 'galaxy_redshift', 'COMPLETENESS_CORRECTION', 'SELECTION_MEMBERSHIP', 'J_ABS_MAG',
            'CLUSTER_AGN']

        # For cluster objects we need to use rejection sampling to induce the inhomogeneous spatial Poisson rate
        if cluster:
            # First, find the projected radial distance from the cluster center for each object (in r500 units)
            agn_skycoords = SkyCoord.from_pixel(sub_catalog['x_pixel'], sub_catalog['y_pixel'],
                                                wcs=cluster_data.mask_wcs)
            radial_distance = cluster_data.sz_center.separation(agn_skycoords).to(u.arcmin)
            radial_distance *= (self._cosmo.kpc_proper_per_arcmin(cluster_data.z_cl).to(u.Mpc / u.arcmin)
                                / cluster_data.r500_cl)

            # Calculate the model rate at each object's location.
            rate_at_radius = self._model_rate(params=input_params,
                                              z=cluster_data.z_cl,
                                              m500=cluster_data.m500_cl,
                                              r500=cluster_data.r500_cl,
                                              r_r500=radial_distance,
                                              j_mag=cluster_data.j_grid)

            # Our rejection rate is defined as the model rate at each object's radius scaled by the maximum rate
            rejection_prob = rate_at_radius / max_rate

            # Draw a random number for each object
            alpha = self._rng.uniform(0., 1., size=rate_at_radius.size)

            # Apply the rejection sampling
            sub_catalog = sub_catalog[rejection_prob >= alpha]

        return sub_catalog

    def _poisson_point_process(self,
                               rate: float,
                               dx: int,
                               dy: int = None,
                               lower_dx: int = 0,
                               lower_dy: int = 0) -> NDArray[float]:
        """
        Uses a spatial Poisson point process to generate AGN candidate coordinates.

        Parameters
        ----------
        rate
            The model rate used in the Poisson distribution to determine the number of points being placed.
        dx, dy
            Upper bound on x- and y-axes respectively. If only `dx` is provided then `dy` = `dx`.
        lower_dx, lower_dy
            Lower bound on x- and y-axes respectively. If not provided, a default of 0 will be used

        Returns
        -------
        NDArray[float]
            Numpy array of (x, y) coordinates of AGN candidates
        """

        # Set dy = dx if not provided
        if dy is None:
            dy = dx

        # Draw from Poisson distribution to determine how many points we will place.
        p = stats.poisson(rate * np.abs(dx - lower_dx) * np.abs(dy - lower_dy)).rvs(random_state=self._rng)

        # Drop `p` points with uniform x and y coordinates
        x = self._rng.uniform(lower_dx, dx, size=p)
        y = self._rng.uniform(lower_dy, dy, size=p)

        # Combine the x and y coordinates.
        coord = np.vstack((x, y))

        return coord

    def _model_rate(self,
                    params: tuple[float, ...],
                    z: float,
                    m500: u.Quantity,
                    r500: u.Quantity,
                    r_r500: NDArray[float] | u.Quantity,
                    j_mag: NDArray[float]) -> NDArray[float]:

        # Unpack our parameters
        theta, eta, zeta, beta, rc = params

        # Luminosity function number
        LF = (self._cosmo.angular_diameter_distance(z) ** 2 * r500 *
              np.trapz(self._luminosity_function(abs_mag=j_mag, redshift=z), j_mag, axis=0))

        # Our amplitude is determined from the cluster data
        a = theta * (1 + z) ** eta * (m500 / (1e15 * u.Msun)) ** zeta * LF

        # Our model rate is a surface density of objects in angular units
        # (as we only have the background in angular units)
        model = a * (1 + (r_r500 / rc) ** 2) ** (-1.5 * beta + 0.5)

        return model  # .value

    def _luminosity_function(self,
                             abs_mag: NDArray[float],
                             redshift: float) -> u.Quantity:
        """
            Assef+11 QLF using luminosity and density evolution.

            Parameters
            ----------
            abs_mag
                Rest-frame J-band absolute magnitude.
            redshift
                Cluster redshift

            Returns
            -------
            Phi
                Luminosity density
            """

        # L/L_*(z) = 10**(0.4 * (M_*(z) - M))
        L_L_star = 10 ** (0.4 * (self._m_star(redshift) - abs_mag))

        # Phi*(z) = 10**(log(Phi*(z))
        phi_star = 10 ** self._log_phi_star(redshift) * (self._cosmo.h / u.Mpc) ** 3
        # phi_star = 10 ** (-4.53) * (cosmo.h / u.Mpc)**3  # PLE

        # QLF slopes
        alpha1 = -3.35  # alpha in Table 2
        alpha2 = -0.37  # beta in Table 2
        # alpha1 = -3.30  # PLE
        # alpha2 = -1.42  # PLE

        Phi = 0.4 * np.log(10) * L_L_star * phi_star * (L_L_star ** -alpha1 + L_L_star ** -alpha2) ** -1

        return Phi

    def _miscentering(self,
                      sigma_offset: float,
                      true_center: SkyCoord,
                      object_coords: SkyCoord,
                      theta_beam: u.Quantity,
                      theta_core: u.Quantity,
                      xi: float,
                      redshift: float,
                      r500: u.Quantity) -> tuple[SkyCoord, u.Quantity, u.Quantity]:
        """
        Performs miscentering operations.

        Offsets the center within the sigma offset of the positional uncertainty relative to the true cluster center.
        This allows for tests of the sensitivity of the radial parameter posteriors with respect to the accuracy of the
        position of the cluster center.

        Parameters
        ----------
        sigma_offset
            Factor of the cluster positional uncertainty to sample a new center within.
        true_center
            The true cluster center as measured by the SZ signal.
        object_coords
            Coordinates of all objects for which we want to calculate radial distances from the new (offset) center.
        theta_beam
            The beam size of the telescope in the detection band.
        theta_core
            The SZ beta-model scale angular radius of the cluster.
        xi
            The SZ detection significance of the cluster.
        redshift
            The cluster redshift.
        r500
            The R500 radius of the cluster.

        Returns
        -------
        tuple[SkyCoord, u.Quantity, u.Quantity]
            The offset cluster center and corresponding radial distances in both units of arcminutes and cluster R500.
        """

        # Determine the positional uncertainty of the cluster center
        cluster_pos_uncert = np.sqrt(theta_beam ** 2 + theta_core ** 2) / xi

        # Scale the positional uncertainty by the offset factor
        cluster_pos_uncert *= sigma_offset

        # Randomly shift the cluster center away from the true center within the scaled positional uncertainty
        offset_center = self._rng.multivariate_normal((true_center.ra.value, true_center.dec.value),
                                                      cov=np.eye(2) * cluster_pos_uncert.to_value(u.deg) ** 2)
        offset_center_coord = SkyCoord(offset_center[0], offset_center[1], unit=u.deg)

        # Compute the new radial distances relative to the offset cluster center in angular and r500 units
        offset_radial_dist_arcmin = offset_center_coord.separation(object_coords).to(u.arcmin)
        offset_radial_dist_r500 = (offset_radial_dist_arcmin * self._cosmo.kpc_proper_per_arcmin(redshift)
                                   .to(u.Mpc / u.arcmin) / r500)

        return offset_center_coord, offset_radial_dist_arcmin, offset_radial_dist_r500

    @staticmethod
    def _print_catalog_stats(params_true: tuple[float, ...], catalog: Table, phot_incomplete: bool):
        """Generates a printout to the log file of various mock catalog summary statistics."""
        if phot_incomplete:
            catalog = catalog[catalog['COMPLETENESS_REJECT'].astype(bool)]
        number_of_clusters = len(catalog.group_by('SPT_ID').groups.keys)
        total_number = len(catalog)
        total_number_comp_corrected = catalog['COMPLETENESS_CORRECTION'].sum()
        total_number_corrected = np.sum(catalog['COMPLETENESS_CORRECTION'] * catalog['SELECTION_MEMBERSHIP'])
        number_per_cluster = total_number_corrected / number_of_clusters
        cluster_objs_corrected = np.sum(catalog['COMPLETENESS_CORRECTION'][catalog['CLUSTER_AGN'].astype(bool)])
        background_objs_corrected = np.sum(catalog['COMPLETENESS_CORRECTION'][~catalog['CLUSTER_AGN'].astype(bool)])
        median_z = np.median(catalog['REDSHIFT'])
        median_m = np.median(catalog['M500'])

        logging.info(f"""Parameters:\t{params_true}
        Number of clusters:\t{number_of_clusters:,}
        Objects Selected:\t{total_number:,}
        Objects selected (completeness corrected):\t{total_number_comp_corrected:,.2f}
        Objects Selected (comp + membership corrected):\t{total_number_corrected:,.2f}
        Objects per cluster (comp + mem corrected):\t{number_per_cluster:,.2f}
        Cluster Objects (corrected):\t{cluster_objs_corrected:,.2f}
        Background Objects (corrected):\t{background_objs_corrected:,.2f}
        SNR (Cluster objects / √Background objects):\t{cluster_objs_corrected / np.sqrt(background_objs_corrected):,.2f}
        Median Redshift:\t{median_z:.2f}
        Median Mass:\t{median_m:.2e}""")


def main():
    # Set common paths
    targeted_output_dir = Path('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/local_backgrounds/'
                               'eta-zeta_slopes/targeted_snr')
    # Set up logger
    logging.basicConfig(filename=targeted_output_dir / 'local_bkg_targeted_snr4.16_mock.log', level=logging.DEBUG)

    # Set our cosmology
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    # Set up rng
    seed = 3775
    logging.info(f'Using RNG seed: {seed}')

    # Set up the luminosity and density evolution using the fits from Assef+11 Table 2
    z_i = [0.25, 0.5, 1., 2., 4.]
    m_star_z_i = [-23.51, -24.64, -26.10, -27.08]
    # m_star_z_i = [-24.52, -25.16, -25.81, -25.10]  # PLE
    phi_star_z_i = [-3.41, -3.73, -4.17, -4.65, -5.77]

    # For the luminosities read in the filters and SED from which we will perform k-corrections on
    irac_36_filter = SpectralElement.from_file('Data_Repository/filter_curves/Spitzer_IRAC/'
                                               '080924ch1trans_full.txt', wave_unit=u.um)
    irac_45_filter = SpectralElement.from_file('Data_Repository/filter_curves/Spitzer_IRAC/'
                                               '080924ch2trans_full.txt', wave_unit=u.um)
    flamingos_j_filter = SpectralElement.from_file('Data_Repository/filter_curves/KPNO/KPNO_2.1m/FLAMINGOS/'
                                                   'FLAMINGOS.BARR.J.MAN240.ColdWitness.txt', wave_unit=u.nm)
    qso2_sed = SourceSpectrum.from_file('Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                        wave_unit=u.Angstrom, flux_unit=units.FLAM)

    # Read in the purity color file
    with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
        sdwfs_purity_data = json.load(f)
    z_bins = sdwfs_purity_data['redshift_bins'][:-1]
    threshold_bins = sdwfs_purity_data['purity_90_colors'][:-1]

    # Read in the SNR-theta fit library
    with open('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/local_backgrounds/eta-zeta_slopes/'
              'snr_to_theta_fits.json', 'r') as f:
        snr_theta_fits = json.load(f)
    for name, fit_data in snr_theta_fits.items():
        snr_theta_fits[name] = Polynomial(fit_data)

    # Read in the SDWFS IRAGN catalog to use to populate photometric information from
    sdwfs_agn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')

    # Read in the local background data
    local_bkgs = QTable.read('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/SPTcl-local_bkg.fits')

    # To avoid issues downstream convert the surface density to arcmin^-2 units
    local_bkgs['LOCAL_BKG_SURF_DEN'] = local_bkgs['LOCAL_BKG_SURF_DEN'].to_value(u.arcmin ** -2)

    # Bin the local backgrounds by color threshold bins
    local_bkgs_grp = local_bkgs.group_by('COLOR_THRESHOLD')
    local_bkgs_mean = local_bkgs_grp['LOCAL_BKG_SURF_DEN'].groups.aggregate(np.mean)
    local_bkgs_std = local_bkgs_grp['LOCAL_BKG_SURF_DEN'].groups.aggregate(np.std)

    # Read in the SPT cluster catalog. We will use real data to source our mock cluster properties.
    bocquet = Table.read('Data_Repository/Catalogs/SPT/SPT_catalogs/2500d_cluster_sample_Bocquet18.fits')

    # For the 20 common clusters between SPT-SZ 2500d and SPTpol 100d surveys we want to update the cluster information
    # from the more recent survey. Thus, we will merge the SPT-SZ and SPTpol catalogs together.
    huang = Table.read('Data_Repository/Catalogs/SPT/SPT_catalogs/sptpol100d_catalog_huang19.fits')

    # First we need to rename several columns in the SPTpol 100d catalog to match the format of the SPT-SZ catalog
    huang.rename_columns(['Dec', 'xi', 'theta_core', 'redshift', 'redshift_unc'],
                         ['DEC', 'XI', 'THETA_CORE', 'REDSHIFT', 'REDSHIFT_UNC'])

    # Now, merge the two catalogs
    sptcl = join(bocquet, huang, join_type='outer')
    sptcl.sort(keys=['SPT_ID', 'field'])  # Sub-sorting by 'field' puts huang entries first
    sptcl = unique(sptcl, keys='SPT_ID', keep='first')  # Keeping huang entries over bocquet
    sptcl.sort(keys='SPT_ID')  # Resort by ID.

    # Convert masses to [Msun] rather than [Msun/1e14]
    sptcl['M500'] *= 1e14 * u.Msun
    sptcl['M500_uerr'] *= 1e14 * u.Msun
    sptcl['M500_lerr'] *= 1e14 * u.Msun

    # Remove any unconfirmed clusters
    sptcl = sptcl[sptcl['M500'] > 0.0]

    # Remove the clusters with z < 0.2 as we have problems assigning photometric data from SDWFS in the redshift bin
    sptcl = sptcl[sptcl['REDSHIFT'] >= 0.2]

    # Add the r500 radius for each cluster
    sptcl['R500'] = (3 * sptcl['M500'] / (4 * np.pi * 500 * cosmo.critical_density(sptcl['REDSHIFT'])
                                          .to(u.Msun / u.Mpc ** 3))) ** 1 / 3

    # For our masks, we will co-op the masks for the real clusters.
    masks_files = [*glob.glob('Data_Repository/Project_Data/SPT-IRAGN/Masks/SPT-SZ_2500d/*.fits'),
                   *glob.glob('Data_Repository/Project_Data/SPT-IRAGN/Masks/SPTpol_100d/*.fits')]

    # Number of clusters to generate
    n_cl = 308

    # Using our targeted SNR, determine the cluster amplitude parameter needed.
    target_snr = 4.16

    # Set the rest of the cluster parameter values
    # theta_range = np.arange(0., 6., 0.1)  # Amplitude
    eta_range = [-5., -3., 0., 3., 4., 5.]  # Redshift slope
    zeta_range = [-2., -1, 0., 1., 2.]  # Mass slope
    beta_true = 1.0  # Radial slope
    rc_true = 0.1  # Core radius (in r500)

    # Set the maximum radius we will generate objects to as a factor of r500
    max_radius = 5.0

    # SPT's 150 GHz beam size
    sz_theta_beam = 1.2 * u.arcmin

    # Set 4.5 um magnitude bounds to define luminosity range
    faint_end_45_apmag = 17.48  # Vega mag
    bright_end_45_apmag = 10.45  # Vega mag

    # Set cluster redshift error for color--redshift selection
    delta_z = 0.1

    # List the miscentering offsets to generate
    miscentering_offsets = [1., 0.75, 0.5]

    # Initialize our mock catalog generator
    mock_catalog_generator = MockIRAGN(cosmo=cosmo, rng_seed=seed,
                                       lf_z=z_i, lf_m_star=m_star_z_i, lf_phi_star=phi_star_z_i,
                                       lf_36_filter=irac_36_filter, lf_45_filter=irac_45_filter,
                                       lf_j_filter=flamingos_j_filter, lf_sed=qso2_sed,
                                       redshift_bins=z_bins, color_thresholds=threshold_bins,
                                       galaxy_data_catalog=sdwfs_agn, cluster_data_catalog=sptcl,
                                       local_bkg_colors=local_bkgs_grp.groups.keys['COLOR_THRESHOLD'],
                                       local_bkg_mean=local_bkgs_mean, local_bkg_std=local_bkgs_std)

    # Create a targeted SNR mock catalog suite
    logging.info(f'Generating targeted SNR = {target_snr} mock catalog suite.')
    mock_catalog_generator.make_targeted_suite(num_clusters=n_cl, mask_files=masks_files,
                                               snr_target=target_snr, snr_theta=snr_theta_fits,
                                               eta_range=eta_range, zeta_range=zeta_range,
                                               beta_true=beta_true, rc_true=rc_true,
                                               max_radius=max_radius,
                                               irac_45_faint_mag=faint_end_45_apmag,
                                               irac_45_bright_mag=bright_end_45_apmag,
                                               cluster_delta_z=delta_z,
                                               sz_theta_beam=sz_theta_beam, miscentering_offsets=miscentering_offsets,
                                               output_dir=targeted_output_dir)


if __name__ == '__main__':
    main()
