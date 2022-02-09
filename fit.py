import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HEADASOUTPUT"] = "/dev/null"

import json
import logging
import warnings
from copy import deepcopy
from itertools import count
from pathlib import Path

import bxa.sherpa as bxa
import numpy as np
import sherpa.astro.ui as shp
from scipy.stats import ks_2samp

import logs
import models
import plots
import stats
import utils
from fluxes import calc_flux_dist, calc_lumin_dist
from priors import BXAPrior
from source import Source
from xmm_backgrounds import get_pn_bkg_model_cached, get_mos_bkg_model_cached


data_path = Path(".", "data_test")


def _set_output_path(srcid, model_name=None):
    folder_name = f"newbxafit_{model_name}" if model_name else "newbxafit"
    return data_path.joinpath(srcid, folder_name)


def _check_fit_exists(output_path):
    results_json_path = output_path.joinpath("info", "results.json")
    return results_json_path.exists()


def _set_sherpa_env(stat="cstat"):
    shp.clean()
    shp.set_stat(stat)
    shp.set_xsabund('wilm')
    shp.set_xsxsect('vern')

    plots.matplotlib_settings()


def _set_obs_prefix(coadd):
    if coadd:
        return "coadd"
    else:
        return "0*"


def _load_data(srcid, coadd=False, emin=0.5, emax=10.0):
    source_path = data_path.joinpath(srcid)

    obs_prefix = _set_obs_prefix(coadd)

    id = 1
    for obsid_path in source_path.glob(obs_prefix):
        for spec_path in obsid_path.glob(f"*SRSPEC{srcid}.FTZ"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shp.load_pha(id, str(spec_path))

            shp.ungroup(id)
            shp.notice_id(id)

            if coadd:
                shp.subtract(id)

            shp.ignore_id(id, lo=None, hi=emin)
            shp.ignore_id(id, lo=emax, hi=None)
            id += 1

    ids = shp.list_data_ids()

    if not ids:
        raise Exception("No data!!!")

    return ids


def _background_models(ids, galabs):
    bkgmodels = []
    for id in ids:
        detector = utils.get_detector(id)

        if detector == "EPN":
            get_epic_bkg_model = get_pn_bkg_model_cached
        else:
            get_epic_bkg_model = get_mos_bkg_model_cached

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bkgmodels.append(get_epic_bkg_model(id, galabs))

    goodness_stats_bkg = _goodness(ids, bkg=True)
    _check_goodness_bkgfit(goodness_stats_bkg)

    return bkgmodels, goodness_stats_bkg


def _set_interinstrument_normalization(ids):
    iin_list = []
    for id in ids[1:]:
        iin_id = shp.xsconstant(f"iin_{id}")
        iin_id.factor.min = 0
        iin_id.factor.max = 2

        iin_list.append(iin_id)

    return iin_list


def _set_full_model_src_bkg(ids, galabs, srcmodel, bkgmodels, iin_list):
    for i, id in enumerate(ids):
        rsp = shp.get_response(id)

        if i > 0 and iin_list:
            srcmodel_id = rsp(iin_list[i - 1] * galabs * srcmodel)
        else:
            srcmodel_id = rsp(galabs * srcmodel)

        shp.set_full_model(id, srcmodel_id + bkgmodels[i])  # bkg_scale included in bkgmodel

            
def _set_full_model_bkg(ids, bkgmodels):
    for i, id in enumerate(ids):
        shp.set_full_model(id, bkgmodels[i])  # bkg_scale included in bkgmodel


def _set_full_model(ids, *args):
    if len(args) == 1:
        _set_full_model_bkg(ids, *args)
    else:
        _set_full_model_src_bkg(ids, *args)


def _check_parameters_different_from_priors(samples, priors, parameters, limit=0.005):
    rng = np.random.default_rng()
    for i, par, prior in zip(count(), parameters, priors):
        prior_sample = [prior(s) for s in rng.random(samples.shape[0])]
        _, pvalue = ks_2samp(samples[:, i], prior_sample)

        if pvalue >= limit:
            logging.warn(
                f"Parameter {par.name} is like prior. "
                f"KS test p-value: {pvalue:0.3f}"
            )


def _fit_all(solver, resume, verbose, **kwargs):
    speed = "safe" # 2*len(parameters)
    results = solver.run(
        resume=resume, verbose=verbose, speed=speed, Lepsilon=0.5, **kwargs
    )
    best_fit_values = [p.val for p in solver.parameters]

    return results["samples"], best_fit_values


def _results_dir_structure(output_path):
    dirs = ["plots", "info", "extra"]
    paths = {}

    for d in dirs:
        d_path = output_path / d
        if not d_path.exists():
            d_path.mkdir(parents=True)

        paths[d] = d_path

    return paths


def _samples_to_ndarray(samples):
    min_chain_size = min([len(s) for s in samples])
    samples_new = np.zeros((min_chain_size, len(samples)))
    for i, s in enumerate(samples):
        samples_new[:, i] = s[min_chain_size-1::-1, 0]

    return samples_new


def _get_evidence(output_path):
    json_path = output_path.joinpath("info", "results.json")
    return json.load(json_path.open())['logz']


def _save_total_evidence(logZ, output_path):
    json_path = output_path.joinpath("results.json")
    with json_path.open("w") as fp:
        json.dump({"logz": logZ}, fp)


def _fit_independent(ids, parameters, output_path, resume, verbose, **kwargs):
    paths = _results_dir_structure(output_path)

    logZ = 0
    samples, best_fit_values = [], []

    for i, id in enumerate(ids):
        ioutput = output_path / f"bkgid{id}"
        iprior = BXAPrior("background_only", [parameters[i]])
        isolver = bxa.BXASolver(
            id=id,
            prior=iprior.function,
            parameters=[parameters[i]],
            outputfiles_basename=ioutput.as_posix(),
        )
        results = isolver.run(resume=resume, verbose=verbose, **kwargs)
        samples.append(results["samples"])
        best_fit_values.append(parameters[i].val)
        logZ += _get_evidence(ioutput)

    samples = _samples_to_ndarray(samples)
    _save_total_evidence(logZ, paths["info"])

    return samples, best_fit_values


def _run_bxa_fit(
    ids, model_name, parameters, zcdf, output_path, resume, verbose, iin_list, **kwargs
):
    prior = BXAPrior(model_name, parameters, zcdf, iin_list)

    solver = bxa.BXASolver(
        id=ids[0],
        otherids=ids[1:],
        prior=prior.function,
        parameters=parameters,
        outputfiles_basename=output_path.as_posix(),
    )

    if model_name != "background_only":
        samples, best_fit_values = _fit_all(solver, resume, verbose, **kwargs)
    else:
        samples, best_fit_values = _fit_independent(
            ids, parameters, output_path, resume, verbose, **kwargs
        )

    _check_parameters_different_from_priors(samples, prior.list, parameters)

    gof = _goodness(ids)

    if model_name != "background_only":
        ppp = _goodness_ppp(ids, parameters, samples, output_path=output_path)
        gof.update(ppp)

    solver.set_best_fit()

    return samples, best_fit_values, solver, gof


def _get_model_disp(id, samples, solver, percentiles, model_bkg_scaled, filename):
    model_disp_path = Path(solver.outputfiles_basename, "extra", filename)

    if not model_disp_path.exists():
        model_disp = utils.calc_model_dispersion(
            id, samples, solver.parameters, model_bkg_scaled, percentiles
        )
        np.savetxt(model_disp_path, model_disp)
    else:
        model_disp = np.loadtxt(model_disp_path)

    return model_disp


def _get_data_for_plots(ids, samples, solver, model_name, best_fit_values, coadd=False):
    # percentiles = [15.87, 50, 84.13]  # -1sigma, median, 1sigma
    percentiles = [2.3, 50, 97.7]  # -2sigma, median, 2sigma
    data_for_plots = []

    for i, id in enumerate(ids):
        shp.notice_id(id)
        shp.ungroup(id)

        data_for_plots.append({})
        current = data_for_plots[i]

        current["model"] = deepcopy(shp.get_fit_plot(id).modelplot)
        if coadd:
            model_bkg_scaled = np.zeros_like(current["model"].y)
            disp_filename = f"model_disp_coadd{id}.dat"
        else:
            current["model_bkg"] = deepcopy(shp.get_bkg_fit_plot(id).modelplot)
            current["backscale"] = shp.get_bkg_scale(id)
            model_bkg_scaled = current["backscale"] * current["model_bkg"].y
            disp_filename = f"model_disp_{id}.dat"

        model_disp = _get_model_disp(
            id, samples, solver, percentiles, model_bkg_scaled, disp_filename
        )
        current["model_disp"] = model_disp[:, :3]
        current["model_src_disp"] = model_disp[:, 3:]

        utils.set_model_to_values(solver.parameters, best_fit_values)
        utils.group_snr_dataset(id, snr=3)

        current["data"] = deepcopy(shp.get_fit_plot(id).dataplot)
        current["resid"] = deepcopy(shp.get_resid_plot(id))
        if not coadd:
            current["data_bkg"] = deepcopy(shp.get_bkg_fit_plot(id).dataplot)
            current["resid_bkg"] = deepcopy(shp.get_bkg_resid_plot(id))

        if model_name == "background_only":
            current["backscale"] = 0

    return data_for_plots


def _get_degrees_of_freedom(ids, data_points, bkg=False):
    degrees_of_freedom = data_points

    for id in ids:
        if bkg:
            # When the background fitting process finish, all parameters are frozen,
            # so we count all parameters in the model to calculate the dof
            m = shp.get_bkg_model(id)
            free_parameters = m.pars
        else:
            m = shp.get_model(id)
            free_parameters = [p for p in m.pars if not p.frozen]

        degrees_of_freedom -= len(free_parameters)

    return degrees_of_freedom


def _calc_cstat(ids, data, model, bkg=False):
    dof = _get_degrees_of_freedom(ids, len(data), bkg=bkg)
    cstat = stats.cstat(data, model)

    return {"cstat": cstat, "dof": dof}


def _goodness(ids, bkg=False, add_cstat=True, add_ppp=True):
    model_total, data_total = [], []
    for id in ids:
        data_counts, model_counts = utils.get_data_model_counts(id, bkg=bkg)
        model_total = np.concatenate((model_total, model_counts))
        data_total = np.concatenate((data_total, data_counts))

    gof = stats.goodness(data_total, model_total)

    if add_cstat:
        gof.update(_calc_cstat(ids, data_total, model_total, bkg=bkg))

    return gof


def _goodness_ppp(ids, parameters, samples, nsims=1000, output_path=None):
    logging.info(
        "Calculating posterior predictive p-value. "
        f"Running {nsims} simulations..."
    )
    ppp = stats.posterior_predictive_pvalue(
        ids, parameters, samples, nsims=nsims, output_path=output_path
    )

    return {"ppp": ppp}


def _check_goodness_bkgfit(gof, limit=0.1):
    pvalue = gof["KS"]["p-value"]

    if  pvalue < limit:
        logging.warn(
            "Background model rejected with 90% confidence. "
            f"KS test p-value: {pvalue:0.3f}"
        )


def _save_goodness_results(gof, gof_bkg, output_path):
    info_path = output_path / "info"
    if not info_path.exists():
        info_path.mkdir()

    json_path = info_path.joinpath("goodness.json")

    with json_path.open("w") as fp:
        json.dump({"src": gof, "bkg": gof_bkg}, fp, indent=2)


def _set_energy_bands():
    return [
        [0.5, 2.0],
        [2.0, 10.0]
    ]


def _get_src_flux(samples, solver, model, ebands):
    flux_dist_path = Path(solver.outputfiles_basename, "extra", "src_flux_dist.dat")

    if not flux_dist_path.exists():
        log_flux_dist_cgs = calc_flux_dist(samples, solver.parameters, model, ebands)
        np.savetxt(flux_dist_path, log_flux_dist_cgs)
        solver.set_best_fit()
    else:
        log_flux_dist_cgs = np.loadtxt(flux_dist_path)

    return log_flux_dist_cgs


def _get_src_lumin(samples, solver, model, ebands, z):
    lumin_dist_path = Path(solver.outputfiles_basename, "extra", "src_lumin_dist.dat")

    if not lumin_dist_path.exists():
        log_lumin_dist_cgs = calc_lumin_dist(samples, solver.parameters, model, ebands, z)
        np.savetxt(lumin_dist_path, log_lumin_dist_cgs)
        solver.set_best_fit()
    else:
        log_lumin_dist_cgs = np.loadtxt(lumin_dist_path)

    return log_lumin_dist_cgs


def _coadded_spectra(srcid, galabs, srcmodel, samples, solver, model_name):
    shp.clean()
    coadd_ids = _load_data(srcid, coadd=True)

    for id in coadd_ids:
        shp.set_source(id, model=galabs*srcmodel)

    solver.set_best_fit()
    best_fit_values = [p.val for p in solver.parameters]

    data_for_plots_coadd = _get_data_for_plots(
        coadd_ids, samples, solver, model_name, best_fit_values, coadd=True
    )

    plots.coadd_spectra(
        coadd_ids, data_for_plots_coadd, output_path=solver.outputfiles_basename
    )


def _run(src, model_name, output_path, resume_fit, verbose, emin=0.2, emax=12.0, iin=False, **kwargs):
    _set_sherpa_env()
    ids = _load_data(src.id, emin=emin, emax=emax)

    logging.info("Setting models...")
    galabs = models.galactic_absorption(src.coords)
    bkgmodels, goodness_stats_bkg = _background_models(ids, galabs)

    if iin:
        iin_list = _set_interinstrument_normalization(ids)
    else:
        iin_list = []

    if model_name == "background_only":
        srcmodels, parameters = models.background_only(bkgmodels)
        _set_full_model(ids, srcmodels)
    else:            
        srcmodel, parameters = models.get_source_model(model_name, src.z)
        _set_full_model(ids, galabs, srcmodel, bkgmodels, iin_list)
        parameters = parameters + [c.factor for c in iin_list]

    samples, best_fit_values, solver, goodness_stats = _run_bxa_fit(
        ids, model_name, parameters, src.zcdf, output_path, resume_fit, verbose, iin_list, **kwargs
    )
    return

    _save_goodness_results(goodness_stats, goodness_stats_bkg, output_path)

    logging.info("Making plots...")
    plots.qqplots(ids, output_path=output_path)
    plots.parameters(samples, parameters, output_path=output_path)

    data_for_plots = _get_data_for_plots(
        ids, samples, solver, model_name, best_fit_values
    )
    plots.spectra(ids, data_for_plots, output_path=output_path)

    if model_name != "background_only":
        logging.info("Calculating fluxes and luminosities...")
        ebands = _set_energy_bands()
        logflux_dist = _get_src_flux(samples, solver, srcmodel, ebands)
        loglumin_dist = _get_src_lumin(samples, solver, srcmodel, ebands, src.z)

        plots.flux(logflux_dist, ebands, output_path=output_path)
        plots.lumin(loglumin_dist, ebands, output_path=output_path)

        logging.info("Plotting coadded spectra...")
        _coadded_spectra(src.id, galabs, srcmodel, samples, solver, model_name)


def fit_spectra(
    src_row, model_name="torus", scratch=False, resume_fit=True, verbose=False, **kwargs
):
    src = Source(src_row, syspdf_params=(0.21, 0.03))
    output_path = _set_output_path(src.id, model_name)

    logs.set_logger(src.id, model_name, stdout_to_log=False)

    if _check_fit_exists(output_path) and not scratch:
        logging.info("Fit available, skipping source.")
    else:
        try:
            _run(src, model_name, output_path, resume_fit, verbose, **kwargs)

        except Exception as e:
            logs.log_exception(e)
            logging.error(f"Fitting source {src.id} failed!")
