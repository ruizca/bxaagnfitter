from pathlib import Path

import sherpa.astro.ui as shp
from gdpyc import GasMap
from sherpa.astro.instrument import RSPModelNoPHA
from sherpa.astro.xspec import read_xstable_model, XScflux
from sherpa.models.parameter import Parameter


models_path = Path(".", "models")


def galactic_absorption(coords):
    nh = GasMap.nh(coords, nhmap="LAB")

    galabs = shp.xstbabs.galabs
    galabs.nH = nh.value / 1e22
    galabs.nH.freeze()

    return galabs


def background_only(bkgmodels):
    srcmodel = []
    parameters = []

    for i, bkg in enumerate(bkgmodels, 1):
        # diagonals rmf and arf
        rmf = bkg.parts[1].parts[0].rmf
        arf = bkg.parts[1].parts[0].arf

        norm = RSPModelNoPHA(arf, rmf, shp.xsconstant(f"bkgrenorm_{i}"))
        srcmodel.append(norm * bkg)

        lognorm = Parameter(
            modelname="src",
            name=f"log_bkgrenorm_{i}",
            val=0,
            min=-2,
            max=2,
            hard_min=-10,
            hard_max=10,
        )
        shp.link(f"bkgrenorm_{i}.factor", 10 ** lognorm)
        parameters.append(lognorm)

    return srcmodel, parameters


def _get_model_powerlaw(redshift):
    intabs = shp.xsztbabs.intabs
    po = shp.xszpowerlw.po
    cflux = XScflux()
    srcmodel = cflux(intabs * po)

    lognH = Parameter(
        modelname="src", name="logNH", val=22, min=20, max=26, hard_min=20, hard_max=26
    )
    intabs.nh = 10 ** (lognH - 22)

    po.PhoIndex.min = 0
    po.PhoIndex.max = 6

    # lognorm = Parameter(
    #     modelname="src", name="lognorm", val=0, min=-8, max=3, hard_min=-20, hard_max=20
    # )
    # lognorm = Parameter(
    #     modelname="src", name="lognorm", val=-4, min=-20, max=20, hard_min=-20, hard_max=20
    # )
    # po.norm = 10 ** lognorm
    po.norm.val = 1
    po.norm.freeze()
    cflux.Emin.val = 0.5
    cflux.Emax.val = 10.0
    cflux.lg10Flux.min = -17
    cflux.lg10Flux.max = -7

    intabs.redshift = po.redshift

    parameters = [
        lognH,
        po.PhoIndex,
        #lognorm,
        cflux.lg10Flux
    ]

    if redshift < 0:
        po.redshift.thaw()
        parameters += [po.redshift]
    else:
        po.redshift.val = redshift
        po.redshift.freeze()

    return srcmodel, parameters


def _get_model_torus(redshift):
    torus = read_xstable_model(
        "torus", models_path.joinpath("uxclumpy-cutoff.fits").as_posix()
    )
    scattering = read_xstable_model(
        "scattering", models_path.joinpath("uxclumpy-cutoff-omni.fits").as_posix()
    )
    srcmodel = torus + scattering

    lognH = Parameter(
        modelname="src", name="logNH", val=22, min=20, max=26, hard_min=20, hard_max=26
    )
    torus.nh = 10 ** (lognH - 22)
    scattering.nh = torus.nh

    # the limits correspond to fluxes between Sco X-1 and CDFS7Ms faintest fluxes
    torus_lognorm = Parameter(
        modelname="src",
        name="torus_lognorm",
        val=0,
        min=-8,
        max=3,
        hard_min=-20,
        hard_max=20,
    )
    torus.norm = 10 ** torus_lognorm

    scattering_lognorm = Parameter(
        "src", "scattering_lognorm", val=-2, min=-7, max=-1, hard_min=-7, hard_max=-1
    )
    scattering.norm = 10 ** (torus_lognorm + scattering_lognorm)
    # shp.set_par("scattering.norm", val=0, frozen=True)

    # Edge-on torus
    # shp.set_par("torus.theta_inc", val=45, frozen=False)
    # torus.theta_inc.val = 85

    scattering.phoindex = torus.phoindex
    scattering.ecut = torus.ecut
    scattering.theta_inc = torus.theta_inc
    scattering.torsigma = torus.torsigma
    scattering.ctkcover = torus.ctkcover
    scattering.redshift = torus.redshift
    scattering.redshift.max = 10

    parameters = [
        lognH,
        torus.phoindex,
        torus.theta_inc,
        torus_lognorm,
        scattering_lognorm,
    ]

    torus.redshift.min = 0
    torus.redshift.max = 10

    if redshift < 0:
        torus.redshift.thaw()
        parameters += [torus.redshift]
    else:
        torus.redshift.val = redshift
        torus.redshift.freeze()

    return srcmodel, parameters


def get_source_model(model, *args, **kwargs):
    try:
        model = globals()[f"_get_model_{model}"]

    except KeyError:
        raise ValueError(f"Model '{model}' is not defined!!!")

    return model(*args, **kwargs)
