"""
SPT_AGN_emcee_sampler_MPI.py
Author: Benjamin Floyd

This script will preform the Bayesian analysis on the SPT-AGN data to produce the posterior probability distributions
for all fitting parameters.
"""
import json
import os
import re
from argparse import ArgumentParser
from time import time

import astropy.units as u
import emcee
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from numpy.typing import NDArray
from schwimmbad import MPIPool
from scipy.interpolate import lagrange, interp1d

from .utils.custom_math import trap_weight


def luminosity_function(abs_mag: NDArray, redshift: NDArray) -> NDArray:
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
    L_L_star = 10 ** (0.4 * (m_star(redshift) - abs_mag))

    # Phi*(z) = 10**(log(Phi*(z))
    phi_star = 10 ** log_phi_star(redshift) * (cosmo.h / u.Mpc) ** 3

    # QLF slopes
    alpha1 = -3.35  # alpha in Table 2
    alpha2 = -0.37  # beta in Table 2

    Phi = 0.4 * np.log(10) * L_L_star * phi_star * (L_L_star ** -alpha1 + L_L_star ** -alpha2) ** -1

    return Phi


def model_rate_opted(params: tuple[float, ...],
                     cluster_id: str,
                     r_r500: NDArray,
                     j_mag: NDArray,
                     integral: bool = False) -> NDArray:
    """
    Our generating model. It is used as the intensity function of our Poisson likelihood function.

    Parameters
    ----------
    params
        Tuple of (theta, eta, zeta, beta, rc, C)
    cluster_id
        SPT ID of our cluster in the catalog dictionary
    r_r500
        A vector of radii of objects within the cluster normalized by the cluster's r500
    j_mag
        A vector of J-band absolute magnitudes to be used in the luminosity function
    integral
        Flag indicating if the luminosity function factor of the model should be integrated. Defaults to `False`.

    Returns
    -------
    model
        A surface density profile of objects as a function of radius and luminosity.
    """

    if args.cluster_only:
        # Unpack our parameters
        theta, eta, zeta, beta, rc = params
        # rc = np.exp(ln_rc)

        # Set background parameter to 0
        c0 = 0
    elif args.background_only:
        # Unpack our parameters
        c0, = params

        # Set all other parameters to neutral values
        theta, eta, zeta = [0.] * 3
        beta, rc = 1 / 3., 1.
    else:
        # Unpack our parameters
        theta, eta, zeta, beta, rc, c0 = params
        # eta, zeta, beta, ln_rc, ln_c0 = params

        # Exponentiate all the log-sampled parameters
        # rc, c0 = np.exp([ln_rc, ln_c0])

    # Extract our data from the catalog dictionary
    z = catalog_dict[cluster_id]['redshift']
    m = catalog_dict[cluster_id]['m500']
    r500 = catalog_dict[cluster_id]['r500']
    local_bkg_offset = catalog_dict[cluster_id]['local_bkg_offset']

    # Luminosity function number
    if integral:
        lum_funct_value = np.trapz(luminosity_function(j_mag, z), j_mag)
    else:
        lum_funct_value = luminosity_function(j_mag, z)

    # if args.no_luminosity or args.poisson_only:
    #     LF = 1
    # else:
    LF = cosmo.angular_diameter_distance(z) ** 2 * r500 * lum_funct_value

    # Convert our background surface density from angular units into units of r500^-2
    if args.cluster_only:
        cz = 0.
    else:
        cz = (((c0 + delta_c(z) + local_bkg_offset) / u.arcmin ** 2)
              * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc) ** 2 * r500 ** 2)

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta * LF

    model = a * (1 + (r_r500 / rc) ** 2) ** (-1.5 * beta + 0.5) + cz

    return model.value


# Set our log-likelihood
def lnlike(param: tuple[float, ...]) -> float:
    """
    Log-likelihood function. It is a modified compound spatial Poisson point process comprising of two parts, the
    inhomogeneous cluster term and a homogeneous background term.

    Parameters
    ----------
    param
        MCMC proposal parameters to be passed to the intensity function.

    Returns
    -------
    float
        The probability value of the log-likelihood function evaluated with input parameters.
    """

    lnlike_list = []
    for cluster_id in catalog_dict:
        # Get the good pixel fraction for this cluster
        gpf_all = catalog_dict[cluster_id]['gpf_rall']

        # Get the radial positions of the AGN
        ri = catalog_dict[cluster_id]['radial_r500_maxr']

        # Get the completeness weights for the AGN
        completeness_weight_maxr = catalog_dict[cluster_id]['completeness_weight_maxr']

        # Get the AGN sample degrees of membership
        # if args.no_selection_membership or args.poisson_only:
        #     agn_membership = 1
        # else:
        #     agn_membership = catalog_dict[cluster_id]['agn_membership_maxr']

        # Get the J-band absolute magnitudes
        # j_band_abs_mag = catalog_dict[cluster_id]['j_abs_mag']

        # Get the radial mesh for integration
        rall = catalog_dict[cluster_id]['rall']

        # Get the luminosity mesh for integration
        jall = catalog_dict[cluster_id]['jall']

        # Compute the completeness ratio for this cluster
        if args.no_completeness or args.poisson_only:
            completeness_ratio = 1.
        else:
            completeness_ratio = len(completeness_weight_maxr) / np.sum(completeness_weight_maxr)

        # Compute the model rate at the locations of the AGN.
        ni = model_rate_opted(param, cluster_id, ri, jall, integral=True)

        # Compute the full model along the radial direction.
        # The completeness weight is set to `1` as the model in the integration is assumed to be complete.
        n_mesh = model_rate_opted(param, cluster_id, rall, jall, integral=True)

        # Use a spatial poisson point-process log-likelihood
        cluster_lnlike = (np.sum(np.log(ni * ri)) - completeness_ratio * trap_weight(n_mesh * 2 * np.pi * rall, rall,
                                                                                     weight=gpf_all))

        lnlike_list.append(cluster_lnlike)

    total_lnlike = np.sum(lnlike_list)

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(params: tuple[float, ...]) -> float:
    """
    Log-prior probability function. This is the joint prior probability of all parameter values. All values are assumed
    to be independent of each other.

    Parameters
    ----------
    params
        MCMC proposal parameter values.

    Returns
    -------
    float
        The joint log-prior probability value evaluated with input parameters.
    """

    cluster_lnpriors = []
    for cluster_id in catalog_dict:
        # Set the background hyperparameters
        global_mean = aux_data['bkg_prior_mean']

        # Allow the user to overwrite the fractional error stored in the preprocessing file
        if args.prior_frac_err is None:
            global_frac_err = aux_data['bkg_prior_frac_err']
        else:
            global_frac_err = args.prior_frac_err

        # Get information from the cluster
        z = catalog_dict[cluster_id]['redshift']
        local_bkg_offset = catalog_dict[cluster_id]['local_bkg_offset']

        # Shift the hyperparameters from the global to local values
        h_c = global_mean + delta_c(z) + local_bkg_offset
        h_c_err = h_c * global_frac_err

        # Extract our parameters
        if args.cluster_only:
            theta, eta, zeta, beta, rc = params
            # rc = np.exp(ln_rc)
            c0 = c0_true
        elif args.background_only:
            c0, = params
            # c0 = np.exp(ln_c0)
            # Set other parameters to values that will pass the priors.
            theta, eta, zeta, beta = [0.] * 4
            rc = 0.1  # Cannot be 0., min rc = 0.05
        else:
            theta, eta, zeta, beta, rc, c0 = params

            # Exponentiate all the log-sampled parameters
            # rc, c0 = np.exp([ln_rc, ln_c0])

        # Shift the parameter to local values too
        if not args.cluster_only:
            c_local = c0 + delta_c(z) + local_bkg_offset
        else:
            c_local = c0

        # Define all priors
        if (0.0 <= theta <= 20. and
                -8. <= eta <= 8. and
                -5. <= zeta <= 5. and
                -3. <= beta <= 3. and
                0.05 <= rc <= 0.5 and
                0.0 <= c_local < np.inf):
            theta_lnprior = 0.0
            eta_lnprior = 0.0
            zeta_lnprior = 0.0
            beta_lnprior = 0.0
            rc_lnprior = 0.0
            if args.cluster_only:
                c_lnprior = 0.
            else:
                c_lnprior = -0.5 * np.sum((c_local - h_c) ** 2 / h_c_err ** 2)
        else:
            theta_lnprior = -np.inf
            eta_lnprior = -np.inf
            zeta_lnprior = -np.inf
            beta_lnprior = -np.inf
            rc_lnprior = -np.inf
            c_lnprior = -np.inf

        ln_prior_prob = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior + rc_lnprior + c_lnprior
        cluster_lnpriors.append(ln_prior_prob)

    total_ln_prior_prob = np.sum(cluster_lnpriors)

    return total_ln_prior_prob


# Define the log-posterior probability
def lnpost(params: tuple[float, ...]) -> float:
    """
    Log-Posterior probability function. The marginal posterior probability function.

    Parameters
    ----------
    params
        MCMC proposal parameter values.

    Returns
    -------
    float
        Returns either the value of the log-posterior value evaluated at the input parameters or negative infinity if
        the joint prior evaluated at the proposed parameter values is not finite.
    """

    lp = lnprior(params)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(params)


if __name__ == '__main__':
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    seed = 3775
    rng = np.random.default_rng(seed)

    # Set up the luminosity and density evolution using the fits from Assef+11 Table 2
    z_i = [0.25, 0.5, 1., 2., 4.]
    m_star_z_i = [-23.51, -24.64, -26.10, -27.08]
    phi_star_z_i = [-3.41, -3.73, -4.17, -4.65, -5.77]
    m_star = lagrange(z_i[1:], m_star_z_i)
    log_phi_star = lagrange(z_i, phi_star_z_i)

    hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
    # hcc_prefix = ''

    parser = ArgumentParser(description='Runs MCMC sampler')
    parser.add_argument('--restart',
                        help='Allows restarting the chain in place rather than resetting the chain.',
                        action='store_true')
    parser.add_argument('file', help='Output chain file name', type=str)
    parser.add_argument('chain_name', help='Chain name', type=str)
    parser.add_argument('--preprocessing', help='Preprocessing file name',
                        default='SPTcl_IRAGN_preprocessing.json', type=str)
    parser.add_argument('--no-luminosity', action='store_true',
                        help='Deactivate luminosity dependence in model.')
    parser.add_argument('--no-selection-membership', action='store_true',
                        help='Deactivate fuzzy degree of membership for AGN selection in likelihood function.')
    parser.add_argument('--no-completeness', action='store_true',
                        help='Deactivate photometric completeness correction in likelihood function.')
    parser.add_argument('--poisson-only', action='store_true',
                        help='Use a pure Poisson likelihood function with a model that has no luminosity dependence.')
    parser.add_argument('--prior-frac-err',
                        help='Overwrites the fractional error present in the preprocessing file with the provided '
                             'value.',
                        type=float)
    parser_grp = parser.add_mutually_exclusive_group()
    parser_grp.add_argument('--cluster-only', action='store_true',
                            help='Sample only on cluster objects.')
    parser_grp.add_argument('--background-only', action='store_true',
                            help='Sample only on background objects.')
    args = parser.parse_args()

    # Load in the prepossessing file
    local_dir = ''
    preprocess_file = os.path.abspath(f'{local_dir}{args.preprocessing}')
    with open(preprocess_file, 'r') as f:
        catalog_dict = json.load(f)

    # Pop out the auxiliary data
    aux_data = catalog_dict.pop('aux_data')

    # Set up interpolators
    agn_purity_color = interp1d(aux_data['redshift_bins'], aux_data['color_thresholds'], kind='previous')
    sdwfs_surf_den = interp1d(aux_data['color_thresholds'][:-1], aux_data['sdwfs_surf_dens'], kind='previous')

    # For convenience, set up the function compositions
    def agn_prior_surf_den(redshift: float) -> float:
        return sdwfs_surf_den(agn_purity_color(redshift))


    # Set up an interpolation for the AGN surface density relative to the reference surface density at z = 0
    delta_c = interp1d(aux_data['redshift_bins'],
                       agn_prior_surf_den(aux_data['redshift_bins']) - agn_prior_surf_den(0.),
                       kind='previous')

    # Go through the catalog dictionary and recasting the cluster's mass and r500 to quantities and recast all the
    # list-type data to numpy arrays
    for cluster_id, cluster_info in catalog_dict.items():
        catalog_dict[cluster_id]['m500'] = cluster_info['m500'] * u.Msun
        catalog_dict[cluster_id]['r500'] = cluster_info['r500'] * u.Mpc
        for data_name, data in filter(lambda x: isinstance(x[1], list), cluster_info.items()):
            catalog_dict[cluster_id][data_name] = np.array(data)

    param_pattern = re.compile(r'(?:[tezbCx]|rc)(-*\d+.\d+|\d+)')
    theta_true, eta_true, zeta_true, beta_true, rc_true, c0_true = np.array(param_pattern.findall(args.chain_name),
                                                                            dtype=float)

    # Set up our MCMC sampler.
    # Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
    ndim = 5 if args.cluster_only else (1 if args.background_only else 6)
    # nwalkers = 6 * ndim
    # ndim = 6
    nwalkers = 50

    # Also, set the number of steps to run the sampler for.
    nsteps = int(100_000)

    # We will initialize our walkers in a tight ball near the initial parameter values.
    # theta_pos0 = rng.uniform(low=0.01, high=15., size=nwalkers)
    theta_pos0 = rng.normal(theta_true, 1e-4, size=nwalkers)
    # eta_pos0 = rng.uniform(low=-6., high=6., size=nwalkers)
    eta_pos0 = rng.normal(eta_true, 1e-4, size=nwalkers)
    # zeta_pos0 = rng.uniform(low=-3., high=3., size=nwalkers)
    zeta_pos0 = rng.normal(zeta_true, 1e-4, size=nwalkers)
    # beta_pos0 = rng.uniform(low=-1., high=1., size=nwalkers)
    beta_pos0 = rng.normal(beta_true, 1e-4, size=nwalkers)
    # rc_pos0 = rng.uniform(0.05, high=0.5, size=nwalkers)
    rc_pos0 = rng.normal(rc_true, 1e-4, size=nwalkers)
    c0_pos0 = rng.normal(c0_true, 1e-4, size=nwalkers)
    if args.cluster_only:
        pos0 = np.array([theta_pos0, eta_pos0, zeta_pos0, beta_pos0, rc_pos0]).T
    elif args.background_only:
        pos0 = np.array([c0_pos0]).T
    else:
        pos0 = np.array([theta_pos0, eta_pos0, zeta_pos0, beta_pos0, rc_pos0, c0_pos0]).T

    # Set up the autocorrelation and convergence variables
    autocorr = np.empty(nsteps)
    old_tau = np.inf  # For convergence

    with MPIPool() as pool:
        backend = emcee.backends.HDFBackend(args.file, name=f'{args.chain_name}'
                                                            f'{"_no-LF" if args.no_luminosity else ""}'
                                                            f'{"_no-mu" if args.no_selection_membership else ""}'
                                                            f'{"_no-comp_corr" if args.no_completeness else ""}'
                                                            f'{"_poisson-only" if args.poisson_only else ""}')
        if not args.restart:
            backend.reset(nwalkers, ndim)

        # Stretch move proposal. Manually specified to tune the `a` parameter.
        # moves = emcee.moves.StretchMove(a=2.75)

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, backend=backend, pool=pool)

        # if not args.restart:
        #     # Run the sampler for a short time to establish the likely maximum posterior location.
        #     burnin_state = sampler.run_mcmc(pos0, nsteps=500)
        #
        #     # Find the likely maximum posterior location and update
        #     max_posterior_means = np.mean(burnin_state.coords, axis=0)
        #
        #     # Reinitialize the walkers in a tight ball around the means found in the burn-in phase
        #     pos0 = rng.normal(loc=max_posterior_means, scale=1e-6, size=pos0.shape)
        #
        #     # Reset the sampler to drop the pre-burn in data
        #     # sampler.reset()

        # Run the sampler.
        print('Starting sampler.')
        start_sampler_time = time()

        # Sample up to nsteps.
        for index, sample in enumerate(sampler.sample(pos0, iterations=nsteps)):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far.
            # Using tol = 0 means we will always get an estimate even if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                print(f'Chains have converged. Ending sampler early.\nIteration stopped at: {sampler.iteration}')
                break
            old_tau = tau

    print(f'Sampler runtime: {time() - start_sampler_time:.2f} s')
