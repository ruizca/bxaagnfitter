import bxa.sherpa as bxa
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal


class BXAPrior:
    def __init__(self, model, *args, **kwargs):
        self._priors = self._set_priors(model, *args, **kwargs)
        self._function = self._set_prior_function()

    @property
    def list(self):
        return self._priors

    @property
    def function(self):
        return self._function

    def _set_priors(self, model, parameters, zcdf=None, iin_list=None):
        prior_method = self._get_prior_method(model)
        priors = prior_method(parameters)

        if not model == "background_only":
            priors = self._add_zprior(priors, zcdf)
            priors = self._add_iinpriors(priors, iin_list)

        return priors

    def _get_prior_method(self, model):
        try:
            prior_method = getattr(self, f"_set_priors_{model}")

        except AttributeError:
            raise ValueError(f"Priors for '{model}' are not defined!!!")

        return prior_method
        
    def _set_priors_powerlaw(self, parameters):
        return [
            bxa.create_uniform_prior_for(parameters[0]),               # logNH
            bxa.create_uniform_prior_for(parameters[1]),               
            #bxa.create_gaussian_prior_for(parameters[1], 1.95, 0.15),  # PhoIndex
            bxa.create_uniform_prior_for(parameters[2]),               # log_norm
        ]

    def _set_priors_torus(self, parameters):
        return [
            bxa.create_uniform_prior_for(parameters[0]),
            bxa.create_gaussian_prior_for(parameters[1], 1.95, 0.15),
            bxa.create_uniform_prior_for(parameters[2]),
            bxa.create_uniform_prior_for(parameters[3]),
            bxa.create_uniform_prior_for(parameters[4]),
        ]
    
    def _set_priors_background_only(self, parameters):
        return [bxa.create_uniform_prior_for(p) for p in parameters]

    def _add_zprior(self, priors, zcdf):
        if zcdf is not None:
            zprior = self._inv_zpdf_func(zcdf)
            priors += [zprior]

        return priors

    def _add_iinpriors(self, priors, iin_list):
        if iin_list is not None:
            iinpriors = [bxa.create_uniform_prior_for(iin.factor) for iin in iin_list]
            priors += iinpriors

        return priors

    def _set_prior_function(self):
        return bxa.create_prior_function(priors=self._priors)

    @staticmethod
    def _inv_zpdf_func(zcdf):
        return lambda x: np.interp(x, zcdf[:, 1], zcdf[:, 0])


class BXAPriorFromPosterior:
    def __init__(self, samples, z):
        self._shape = self._get_nbins(z)
        self._cdf, self._parvals = self._calc_cdf(samples)
        self._function = self._set_prior_function()
        
    @property
    def function(self):
        return self._function

    def _calc_cdf(self, samples):
        posterior_pdf, edges = np.histogramdd(samples, density=True, bins=self._shape)
        middle_points =  self._calc_middle_points(edges)
        dV = self._calc_volume_element(edges)
        posterior_cdf = posterior_pdf.cumsum() * dV
        
        win = signal.windows.hann(10)
        posterior_cdf = signal.convolve(posterior_cdf, win, mode='same') / sum(win)

        idxs = np.arange(0, len(posterior_cdf), dtype=int)
        itp = interp1d(
            posterior_cdf,
            idxs,
            kind="nearest",
            fill_value=(0, len(posterior_cdf) - 1),
            bounds_error=False
        )
        
        return itp, middle_points

    def _prior(self, x):
        idx = [np.unravel_index(int(i), self._shape) for i in self._cdf(x)]
        params = np.array(
            [
                [self._parvals[j][i[j]] for j in range(len(self._shape))] for i in idx
            ]
        )
        return params
        
    def _set_prior_function(self):
        return lambda x: self._prior(x)

    @staticmethod
    def _calc_middle_points(edges):
        return [0.5*(e[:-1] + e[1:]) for e in edges]
    
    @staticmethod
    def _calc_volume_element(edges):
        # This only work for regularly spaced bins
        return np.prod([e[1] - e[0] for e in edges])

    @staticmethod
    def _get_nbins(z):
        # This is for torus
        # TODO: use model argument to select the correct binning
        nbins = (30, 20, 10, 30, 30, 35)
        
        if z > 0:
            nbins = nbins[:-1]
            
        return nbins
