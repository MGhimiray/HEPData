# %%
import hepdata_lib
import awkward as ak
import numpy as np
import re
from hepdata_lib import Variable, Uncertainty
from hepdata_lib import RootFileReader
from hepdata_lib import Table
from function import build_hepdata_table_from_plot_root

print("hepdata_lib version", hepdata_lib.__version__)

# %%
NDEC_PLOT = 1
NDEC_UNFOLDED = 3
NDEC_SUMMARY = 2

def round_values(values, ndigits):
    out = []
    for v in values:
        if v is None:
            out.append("-")
        elif isinstance(v, str) and v.strip() in {"", "-", "—", "NA", "N/A", "\\NA"}:
            out.append("-")
        else:
            out.append(round(float(v), ndigits))
    return out


def round_uncertainties(values, ndigits):
    out = []
    for v in values:
        if v is None:
            out.append(None)
        elif isinstance(v, str) and v.strip() in {"", "-", "—", "NA", "N/A", "\\NA"}:
            out.append(None)
        else:
            out.append(round(float(v), ndigits))
    return out



# # create submission

# %%
from hepdata_lib import Submission
submission = Submission()

submission.read_abstract("inputs/abstract.txt")
submission.add_link("Webpage with all figures and tables", "https://cds.cern.ch/record/2957045?ln=en")

# # functions

# %%
def get_error(nominal_hist, error_histlist=None, isdata=False, normalized=False):
    nominal_arr = ak.Array(nominal_hist["y"])
    stat_err = ak.Array(nominal_hist["dy"])

    if isdata:
        nominal_arr = nominal_arr * ak.Array(error_histlist[0]["y"])
        stat_err = stat_err * ak.Array(error_histlist[0]["y"])

    if normalized:
        nominal_sum = ak.sum(nominal_arr)
        if nominal_sum != 0:
            nominal_arr = nominal_arr / nominal_sum
            stat_err = stat_err / nominal_sum

    if isdata:
        return nominal_arr, stat_err, stat_err, stat_err

    safe_nominal = ak.where(nominal_arr != 0, nominal_arr, 1.0)

    dstat = stat_err / safe_nominal
    dstat2 = np.power(dstat, 2)

    dsyst = []
    for i in error_histlist:
        varied = ak.Array(i["y"])
        if normalized:
            varied_sum = ak.sum(varied)
            if varied_sum != 0:
                varied = varied / varied_sum
        dsyst.append(np.abs(1 - varied / safe_nominal))

    dtot_err2 = dstat2
    dsyst_err2 = ak.zeros_like(dstat2)

    for i in dsyst:
        dtot_err2 = dtot_err2 + np.power(i, 2)
        dsyst_err2 = dsyst_err2 + np.power(i, 2)

    tot_err = np.sqrt(dtot_err2) * nominal_arr
    syst_err = np.sqrt(dsyst_err2) * nominal_arr

    return nominal_arr, stat_err, syst_err, tot_err


def get_hist_yaxis(y_info, isdata=False, units="", normalized=False, ndigits=4):
    nominal_arr, stat_err, syst_err, tot_err = get_error(
        y_info["h_nominal"], y_info["h_syst_list"], isdata, normalized
    )

    y_val = Variable(y_info["label"], is_independent=False, is_binned=False, units=units)
    y_val.values = round_values(nominal_arr.to_list(), ndigits)

    unc_stat = Uncertainty("stat", is_symmetric=True)
    unc_stat.values = round_uncertainties(stat_err.to_list(), ndigits)
    y_val.add_uncertainty(unc_stat)

    if not isdata:
        unc_syst = Uncertainty("syst", is_symmetric=True)
        unc_syst.values = round_uncertainties(syst_err.to_list(), ndigits)
        y_val.add_uncertainty(unc_syst)

    return y_val


def get_table(hist_name, hist_map):
    print("Processing {}".format(hist_name), hist_map["normalized"])

    reader_data = RootFileReader(
        "/eos/user/m/mghimira/Combine/unfolding_run3/makePlots/input_files/xs_{}_normalized{}.root".format(
            hist_map["key"], int(hist_map["normalized"])
        )
    )
    reader_sh2 = RootFileReader("/eos/user/m/mghimira/Combine/unfolding_run3/makePlots/histogen_vbsvv_ewk_sherpa.root")
    reader_mg5 = RootFileReader("/eos/user/m/mghimira/Combine/unfolding_run3/makePlots/histogen_vbsvv_ewk_mapgraph_noewkcorr.root")
    reader_mg5_corr = RootFileReader("/eos/user/m/mghimira/Combine/unfolding_run3/makePlots/histogen_vbsvv_ewk_mapgraph.root")

    y_val_dict = {}

    y_val_dict["Data"] = {
        "label": "Data",
        "h_nominal": reader_data.read_hist_1d(f"hD{hist_map['key']}"),
        "h_syst_list": [
            reader_mg5_corr.read_hist_1d(f"hD{hist_map['key']}")
        ],
        "isdata": True,
    }
    y_val_dict["mg5"] = {
        "label": r"MG5_aMC@NLO + Py8 (no NLO corr.)",
        "h_nominal": reader_mg5.read_hist_1d(f"hD{hist_map['key']}"),
        "h_syst_list": [
            reader_mg5.read_hist_1d(f"hD{hist_map['key']}_QCD"),
            reader_mg5.read_hist_1d(f"hD{hist_map['key']}_PDF"),
            reader_mg5.read_hist_1d(f"hD{hist_map['key']}_PS"),
        ],
        "isdata": False,
    }
    y_val_dict["mg5_corr"] = {
        "label": r"MG5_aMC@NLO + Py8 (NLO corr.)",
        "h_nominal": reader_mg5_corr.read_hist_1d(f"hD{hist_map['key']}"),
        "h_syst_list": [
            reader_mg5_corr.read_hist_1d(f"hD{hist_map['key']}_QCD"),
            reader_mg5_corr.read_hist_1d(f"hD{hist_map['key']}_PDF"),
            reader_mg5_corr.read_hist_1d(f"hD{hist_map['key']}_PS"),
        ],
        "isdata": False,
    }
    y_val_dict["sh2"] = {
        "label": r"Sherpa 2",
        "h_nominal": reader_sh2.read_hist_1d(f"hD{hist_map['key']}"),
        "h_syst_list": [
            reader_sh2.read_hist_1d(f"hD{hist_map['key']}_QCD"),
            reader_sh2.read_hist_1d(f"hD{hist_map['key']}_PDF"),
            reader_sh2.read_hist_1d(f"hD{hist_map['key']}_PS"),
        ],
        "isdata": False,
    }

    x_edges = y_val_dict["Data"]["h_nominal"]["x_edges"]
    x_centers = [0.5 * (edge[0] + edge[1]) for edge in x_edges]

    x_bin = Variable(
        hist_map["label"],
        is_independent=True,
        is_binned=False,
        units=hist_map["label_unit"],
    )
    x_bin.values = round_values(x_centers, NDEC_UNFOLDED)

    y_vals = []
    for name in y_val_dict:
        y_vals.append(
            get_hist_yaxis(
                y_val_dict[name],
                isdata=y_val_dict[name]["isdata"],
                units=hist_map["unit"],
                normalized=hist_map["normalized"],
                ndigits=NDEC_UNFOLDED,
            )
        )

    table = Table(hist_name)
    table.location = hist_map["location"]
    table.description = hist_map["description"]
    table.keywords["observables"] = [hist_map["label"]]
    # table.add_image(hist_map["image"])

    table.add_variable(x_bin)
    for y in y_vals:
        table.add_variable(y)

    return table


def get_summary_xs_table():
    processes = [
        r"EW $W^{\pm}W^{\pm}$",
        r"EW+QCD $W^{\pm}W^{\pm}$",
        r"EW WZ",
        r"EW+QCD WZ",
    ]

    meas_val  = [3.81, 4.32, 1.43, 4.75]
    meas_stat = [0.33, 0.36, 0.23, 0.27]
    meas_syst = [0.18, 0.18, 0.12, 0.22]

    mg5_no_val = [4.27, 4.75, 1.45, 4.59]
    mg5_no_unc = [0.38, 0.52, 0.13, 1.07]

    mg5_nlo_val = [3.51, 3.99, 1.25, 4.39]
    mg5_nlo_unc = [0.31, 0.44, 0.11, 1.05]

    sherpa_val = [3.37, "-", 1.24, "-"]
    sherpa_unc = [0.50, None, 0.19, None]

    table = Table("Table 6")
    table.location = "Data from Table 6"
    table.description = (
        r"Measured fiducial cross sections times branching fraction "
        r"($\sigma\mathcal{B}$) and theory predictions for EW and EW+QCD "
        r"$W^{\pm}W^{\pm}$ and WZ production."
    )
    table.keywords["observables"] = [r"$\sigma\mathcal{B}$"]
    table.keywords["phrases"] = ["fiducial cross section", "VBS", "W+W+", "WZ"]

    proc = Variable("Process", is_independent=True, is_binned=False, units="")
    proc.values = processes
    table.add_variable(proc)

    meas = Variable(r"$\sigma\mathcal{B}$", is_independent=False, is_binned=False, units="fb")
    meas.values = round_values(meas_val, NDEC_SUMMARY)

    unc_stat = Uncertainty("stat", is_symmetric=True)
    unc_stat.values = round_uncertainties(meas_stat, NDEC_SUMMARY)
    meas.add_uncertainty(unc_stat)

    unc_syst = Uncertainty("syst", is_symmetric=True)
    unc_syst.values = round_uncertainties(meas_syst, NDEC_SUMMARY)
    meas.add_uncertainty(unc_syst)

    table.add_variable(meas)

    mg5_no = Variable(
        r"MG5_aMC@NLO prediction without NLO corrections",
        is_independent=False,
        is_binned=False,
        units="fb",
    )
    mg5_no.values = round_values(mg5_no_val, NDEC_SUMMARY)

    mg5_no_err = Uncertainty("", is_symmetric=True)
    mg5_no_err.values = round_uncertainties(mg5_no_unc, NDEC_SUMMARY)
    mg5_no.add_uncertainty(mg5_no_err)

    table.add_variable(mg5_no)

    mg5_nlo = Variable(
        r"MG5_aMC@NLO prediction with NLO corrections",
        is_independent=False,
        is_binned=False,
        units="fb",
    )
    mg5_nlo.values = round_values(mg5_nlo_val, NDEC_SUMMARY)

    mg5_nlo_err = Uncertainty(" ", is_symmetric=True)
    mg5_nlo_err.values = round_uncertainties(mg5_nlo_unc, NDEC_SUMMARY)
    mg5_nlo.add_uncertainty(mg5_nlo_err)

    table.add_variable(mg5_nlo)

    sherpa = Variable(
        r"SHERPA prediction without NLO corrections",
        is_independent=False,
        is_binned=False,
        units="fb",
    )
    sherpa.values = round_values(sherpa_val, NDEC_SUMMARY)

    sherpa_err = Uncertainty(" ", is_symmetric=True)
    sherpa_err.values = round_uncertainties(sherpa_unc, NDEC_SUMMARY)
    sherpa.add_uncertainty(sherpa_err)

    table.add_variable(sherpa)

    return table


def get_summary_uncertainty_table(txtfile, as_percent=False):
    sources = []
    ew_ww = []
    ew_wz = []

    scale = 100.0 if as_percent else 1.0
    unit = "%" if as_percent else ""

    with open(txtfile, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.lower().startswith("source uncertainty"):
                continue
            if set(line) <= set("- "):
                continue

            parts = [x.strip() for x in line.split("|")]
            if len(parts) != 3:
                continue

            sources.append(parts[0])
            ew_ww.append(float(parts[1]) * scale)
            ew_wz.append(float(parts[2]) * scale)

    source_var = Variable("Source of uncertainty", is_independent=True, is_binned=False, units="")
    source_var.values = sources

    ww_var = Variable(r"EW $\mathrm{W}^{\pm}\mathrm{W}^{\pm}$", is_independent=False, is_binned=False, units=unit)
    ww_var.values = round_values(ew_ww, NDEC_SUMMARY)

    wz_var = Variable(r"EW $\mathrm{W}^{\pm}\mathrm{Z}$", is_independent=False, is_binned=False, units=unit)
    wz_var.values = round_values(ew_wz, NDEC_SUMMARY)

    table = Table("Table 3")
    table.location = "Data from Table 3"
    table.description = (
        "Systematic uncertainties in the signal strength $\\mu$, defined as the measured inclusive EW cross sections divided by the predicted cross sections. "
        "Lepton experimental uncertainties encompass the effects of the calibration of lepton momentum scale and resolution, as well as lepton trigger, reconstruction, "
        "identification, and isolation efficiencies. Jet experimental uncertainties encompass the effects of JES and JER. "
        "The limited sample size refers to the finite number of MC and data events used to estimate the backgrounds."
    )
    table.keywords["observables"] = ["relative uncertainty"]

    table.add_variable(source_var)
    table.add_variable(ww_var)
    table.add_variable(wz_var)

    return table


def parse_yield_cell(cell):
    cell = cell.strip()

    if cell in ["—", "-", "NA", "N/A", "\\NA"]:
        return None, None

    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(?:±|\+-)\s*([0-9]*\.?[0-9]+)\s*$", cell)
    if m:
        return float(m.group(1)), float(m.group(2))

    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*$", cell)
    if m:
        return float(m.group(1)), None

    raise ValueError(f"Could not parse cell: {cell}")


def get_table4_yields(txtfile):
    with open(txtfile, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = [x.strip() for x in lines[0].split("|")]
    regions = header[1:]

    rows = []
    for line in lines[1:]:
        parts = [x.strip() for x in line.split("|")]
        rows.append(parts)

    process_names = [r[0] for r in rows]

    table = Table("Table 4")
    table.location = "Data from Table 4"
    table.description = (
        "Expected yields from SM processes and observed data events in the "
        "$W^{\pm}W^{\pm}$ signal region, nonprompt lepton control region, WZ signal region, "
        "and tZq control region. The quoted uncertainties are the combination of "
        "statistical and systematic contributions. The expected yields correspond "
        "to the best-fit normalizations from the simultaneous fit to data."
    )
    table.keywords["observables"] = ["event yield"]

    proc_var = Variable("Process", is_independent=True, is_binned=False, units="")
    proc_var.values = process_names
    table.add_variable(proc_var)

    for ireg, region in enumerate(regions, start=1):
        values = []
        uncertainties = []

        for row in rows:
            val, err = parse_yield_cell(row[ireg])

            if val is None:
                values.append("-")
                uncertainties.append(None)
            else:
                values.append(round(float(val), NDEC_SUMMARY))
                uncertainties.append(round(float(err), NDEC_SUMMARY) if err is not None else None)

        y_var = Variable(region, is_independent=False, is_binned=False, units="events")
        y_var.values = values

        unc = Uncertainty("", is_symmetric=True)
        unc.values = uncertainties
        y_var.add_uncertainty(unc)

        table.add_variable(y_var)

    return table


# %%
input_key = {
    'Figure 5a': {
        'key': "EWKWWMJJ",
        'normalized': False,
        'label': r"$m_{jj}$",
        'label_unit': "GeV",
        'unit': "fb/bin",
        'location': "Data from Figure 5a",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_005-a.pdf",
        'description': r"Measured and predicted absolute $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $m_{jj}$",
    },
    'Figure 5b': {
        'key': "EWKWWMLL",
        'normalized': False,
        'label': r"$m_{\ell \ell}$",
        'label_unit': "GeV",
        'unit': "fb/bin",
        'location': "Data from Figure 5b",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_005-b.pdf",
        'description': r"Measured and predicted absolute $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $m_{\ell \ell}$",
    },
    'Figure 5c': {
        'key': "EWKWWDELTAETAJJ",
        'normalized': False,
        'label': r"$\Delta \eta _{jj}$",
        'label_unit': "",
        'unit': "fb/bin",
        'location': "Data from Figure 5c",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_005-c.pdf",
        'description': r"Measured and predicted absolute $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $\Delta \eta _{jj}$",
    },
    'Figure 5d': {
        'key': "EWKWWDELTAPHIJJ",
        'normalized': False,
        'label': r"$\Delta \phi _{jj}$",
        'label_unit': "",
        'unit': "fb/bin",
        'location': "Data from Figure 5d",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_005-d.pdf",
        'description': r"Measured and predicted absolute $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $\Delta \phi _{jj}$",
    },
    'Figure 5e': {
        'key': "EWKWWNJET",
        'normalized': False,
        'label': r"$n_j$",
        'label_unit': "",
        'unit': "fb/bin",
        'location': "Data from Figure 5e",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_005-e.pdf",
        'description': r"Measured and predicted absolute $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $n_j$",
    },
    'Figure 6a': {
        'key': "EWKWWMJJ",
        'normalized': True,
        'label': r"$m_{jj}$",
        'label_unit': "GeV",
        'unit': "1/bin",
        'location': "Data from Figure 6a",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_006-a.pdf",
        'description': r"Measured and predicted normalized $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $m_{jj}$",
    },
    'Figure 6b': {
        'key': "EWKWWMLL",
        'normalized': True,
        'label': r"$m_{\ell \ell}$",
        'label_unit': "GeV",
        'unit': "1/bin",
        'location': "Data from Figure 6b",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_006-b.pdf",
        'description': r"Measured and predicted normalized $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $m_{\ell \ell}$",
    },
    'Figure 6c': {
        'key': "EWKWWDELTAETAJJ",
        'normalized': True,
        'label': r"$\Delta \eta _{jj}$",
        'label_unit': "",
        'unit': "1/bin",
        'location': "Data from Figure 6c",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_006-c.pdf",
        'description': r"Measured and predicted normalized $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $\Delta \eta _{jj}$",
    },
    'Figure 6d': {
        'key': "EWKWWDELTAPHIJJ",
        'normalized': True,
        'label': r"$\Delta \phi _{jj}$",
        'label_unit': "",
        'unit': "1/bin",
        'location': "Data from Figure 6d",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_006-d.pdf",
        'description': r"Measured and predicted normalized $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $\Delta \phi _{jj}$",
    },
    'Figure 6e': {
        'key': "EWKWWNJET",
        'normalized': True,
        'label': r"$n_j$",
        'label_unit': "",
        'unit': "1/bin",
        'location': "Data from Figure 6e",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_006-e.pdf",
        'description': r"Measured and predicted normalized $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{W}^{\pm}$ cross sections in bins of $n_j$",
    },
    'Figure 7a': {
        'key': "EWKWZMJJ",
        'normalized': False,
        'label': r"$m_{jj}$",
        'label_unit': "GeV",
        'unit': "fb/bin",
        'location': "Data from Figure 7a",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_007-a.pdf",
        'description': r"Measured and predicted absolute $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{Z}$ cross sections in bins of $m_{jj}$",
    },
    'Figure 7b': {
        'key': "EWKWZMJJ",
        'normalized': True,
        'label': r"$m_{jj}$",
        'label_unit': "GeV",
        'unit': "1/bin",
        'location': "Data from Figure 7b",
        'image': "/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/Figure_007-b.pdf",
        'description': r"Measured and predicted normalized $\mathrm{EW}~\mathrm{W}^{\pm} \mathrm{Z}$ cross sections in bins of $m_{jj}$",
    },
}

submission.add_table(get_summary_uncertainty_table("inputs/table3.txt", as_percent=False))
yield_table = get_table4_yields("inputs/table4.txt")

table1 = build_hepdata_table_from_plot_root(
    root_path="/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/finalPlots/ssww_sswwAnalysis1006_bin0.root",
    table_name="Figure 3a",
    x_label=r"$m_{jj}$",
    x_unit="GeV",
    y_unit="Events/GeV",
    description="Distribution of $m_{jj}$ in the signal $W^{\pm}W^{\pm}$ region.",
    observable=r"$m_{jj}$",
    location="Data from Figure 3a",
    ndigits=NDEC_PLOT,
)

table2 = build_hepdata_table_from_plot_root(
    root_path="/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/finalPlots/ssww_sswwAnalysis1007_bin0.root",
    table_name="Figure 3b",
    x_label=r"$m_{\ell\ell}$",
    x_unit="GeV",
    y_unit="Events/GeV",
    description="Distribution of $m_{\ell \ell}$ in the signal $W^{\pm}W^{\pm}$ region.",
    observable=r"$m_{\ell \ell}$",
    location="Data from Figure 3b",
    ndigits=NDEC_PLOT,
)

table3 = build_hepdata_table_from_plot_root(
    root_path="/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/finalPlots/ssww_sswwAnalysis1009_bin0.root",
    table_name="Figure 3c",
    x_label=r"$\Delta \eta_{jj}$",
    x_unit="",
    y_unit="Events/1",
    description="Distribution of $\Delta \eta_{jj}$ in the signal $W^{\pm}W^{\pm}$ region.",
    observable=r"$\Delta \eta_{jj}$",
    location="Data from Figure 3c",
    ndigits=NDEC_PLOT,
)

table4 = build_hepdata_table_from_plot_root(
    root_path="/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/finalPlots/ssww_sswwAnalysis1010_bin0.root",
    table_name="Figure 3d",
    x_label=r"$\Delta \phi_{jj}$",
    x_unit="",
    y_unit="Events/1",
    description="Distribution of $\Delta \phi_{jj}$ in the signal $W^{\pm}W^{\pm}$ region.",
    observable=r"$\Delta \phi_{jj}$",
    location="Data from Figure 3d",
    ndigits=NDEC_PLOT,
)

table5 = build_hepdata_table_from_plot_root(
    root_path="/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/finalPlots/ssww_sswwAnalysis1008_bin0.root",
    table_name="Figure 3e",
    x_label=r"$n_j$",
    x_unit="",
    y_unit="Events/bin",
    description="Distribution of $n_j$ in the signal $W^{\pm}W^{\pm}$ region.",
    observable=r"$n_j$",
    location="Data from Figure 3e",
    ndigits=NDEC_PLOT,
)

table6 = build_hepdata_table_from_plot_root(
    root_path="/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/finalPlots/ssww_wzAnalysis1003_bin0.root",
    table_name="Figure 4a",
    x_label="BDT output",
    x_unit="",
    y_unit="Events/bin",
    description="Distribution of BDT output in the signal WZ region.",
    observable="BDT output",
    location="Data from Figure 4a",
    ndigits=NDEC_PLOT,
)

table7 = build_hepdata_table_from_plot_root(
    root_path="/eos/user/m/mghimira/AN-25-031/gitlab/papers/SMP-25-013/finalPlots/ssww_wzAnalysis1004_bin0.root",
    table_name="Figure 4b",
    x_label=r"$m_{jj}$",
    x_unit="GeV",
    y_unit="Events/GeV",
    description="Distribution of $m_{jj}$ in the signal WZ region.",
    observable=r"$m_{jj}$",
    location="Data from Figure 4b",
    ndigits=NDEC_PLOT,
)

submission.add_table(table1)
submission.add_table(table2)
submission.add_table(table3)
submission.add_table(table4)
submission.add_table(table5)
submission.add_table(table6)
submission.add_table(table7)

for i in input_key:
    table_tmp = get_table(i, input_key[i])
    submission.add_table(table_tmp)

submission.add_table(yield_table)
submission.add_table(get_summary_xs_table())

submission.create_files("example_output", validate=True, remove_old=True)