import ROOT
from hepdata_lib import Table, Variable, Uncertainty

K_PLOT_DATA      = 0
K_PLOT_QQWW      = 1
K_PLOT_GGWW      = 2
K_PLOT_TT        = 3
K_PLOT_TW        = 4
K_PLOT_DY        = 5
K_PLOT_EWKSSWW   = 6
K_PLOT_QCDSSWW   = 7
K_PLOT_EWKWZ     = 8
K_PLOT_WZ        = 9
K_PLOT_ZZ        = 10
K_PLOT_NONPROMPT = 11
K_PLOT_VVV       = 12
K_PLOT_TVX       = 13
K_PLOT_VG        = 14
K_PLOT_HIGGS     = 15
K_PLOT_WS        = 16
K_PLOT_OTHER     = 17
K_PLOT_EM        = 18
K_PLOT_BSM       = 19
K_PLOT_SIGNAL0   = 20
K_PLOT_SIGNAL1   = 21
K_PLOT_SIGNAL2   = 22
K_PLOT_SIGNAL3   = 23
K_PLOT_SIGNAL4   = 24
K_PLOT_SIGNAL5   = 25

PLOT_LABELS_SSWW = {
    K_PLOT_NONPROMPT: "Nonprompt",
    K_PLOT_VVV:       "VVV",
    K_PLOT_TVX:       "tVx",
    K_PLOT_VG:        "Vγ",
    K_PLOT_WS:        "Wrong sign",
    K_PLOT_EWKSSWW:   "EW W±W±",
    K_PLOT_QCDSSWW:   "QCD W±W±",
    K_PLOT_EWKWZ:     "EW WZ",
    K_PLOT_WZ:        "QCD WZ",
    K_PLOT_ZZ:        "ZZ",
}

SHOWN_COMPONENTS_SSWW = [
    K_PLOT_NONPROMPT,
    K_PLOT_VVV,
    K_PLOT_TVX,
    K_PLOT_VG,
    K_PLOT_WS,
    K_PLOT_EWKSSWW,
    K_PLOT_QCDSSWW,
    K_PLOT_EWKWZ,
    K_PLOT_WZ,
    K_PLOT_ZZ,
]


def round_values(values, ndigits=1):
    out = []
    for v in values:
        if v is None:
            out.append("-")
        elif isinstance(v, str) and v.strip() in {"", "-", "—", "NA", "N/A", "\\NA"}:
            out.append("-")
        else:
            out.append(round(float(v), ndigits))
    return out


def round_uncertainties(values, ndigits=2):
    out = []
    for v in values:
        if v is None:
            out.append(None)
        elif isinstance(v, str) and v.strip() in {"", "-", "—", "NA", "N/A", "\\NA"}:
            out.append(None)
        else:
            out.append(round(float(v), ndigits))
    return out


def _clone_hist(h, name):
    out = h.Clone(name)
    out.SetDirectory(0)
    return out


def _empty_like(h, name):
    out = _clone_hist(h, name)
    out.Reset("ICES")
    return out


def _hist_to_edges_values_errors(h):
    edges, vals, errs = [], [], []
    for i in range(1, h.GetNbinsX() + 1):
        xlow = float(h.GetBinLowEdge(i))
        xhigh = float(h.GetBinLowEdge(i) + h.GetBinWidth(i))
        vals.append(float(h.GetBinContent(i)))
        errs.append(float(h.GetBinError(i)))
        edges.append((xlow, xhigh))
    return edges, vals, errs


def build_hepdata_table_from_plot_root(
    root_path,
    table_name,
    x_label,
    x_unit="",
    y_unit="Events/bin",
    description="",
    location="",
    observable=None,
    image_path=None,
    clip_negative_bins=True,
    ndigits=3,
):
    f = ROOT.TFile.Open(root_path, "READ")
    if not f or f.IsZombie():
        raise OSError(f"Could not open ROOT file: {root_path}")

    hist = {}
    for ic in range(26):
        h = f.Get(f"histo{ic}")
        if h:
            hist[ic] = _clone_hist(h, f"histo{ic}_clone")

    htotal = f.Get("histo_total")
    htotal = _clone_hist(htotal, "histo_total_clone") if htotal else None
    f.Close()

    if K_PLOT_DATA not in hist:
        raise ValueError("Data histogram histo0 not found.")

    template = hist[K_PLOT_DATA]

    if clip_negative_bins:
        for h in hist.values():
            for i in range(1, h.GetNbinsX() + 1):
                if h.GetBinContent(i) < 0:
                    h.SetBinContent(i, 0.0)
                    h.SetBinError(i, 0.0)

    # merge Signal0-3 into EWKSSWW
    if K_PLOT_EWKSSWW in hist:
        for isig in [K_PLOT_SIGNAL0, K_PLOT_SIGNAL1, K_PLOT_SIGNAL2, K_PLOT_SIGNAL3]:
            if isig in hist and hist[isig].GetSumOfWeights() != 0:
                hist[K_PLOT_EWKSSWW].Add(hist[isig])
                hist[isig].Scale(0.0)

    table = Table(table_name)
    table.description = description
    if location:
        table.location = location
    if observable:
        table.keywords["observables"] = [observable]
    if image_path:
        table.add_image(image_path)

    # x-axis as bin centers only
    x_edges, _, _ = _hist_to_edges_values_errors(template)
    x_centers = [0.5 * (xlow + xhigh) for (xlow, xhigh) in x_edges]

    x_var = Variable(x_label, is_independent=True, is_binned=False, units=x_unit)
    x_var.values = round_values(x_centers, ndigits)
    table.add_variable(x_var)

    # Data
    _, data_vals, data_errs = _hist_to_edges_values_errors(hist[K_PLOT_DATA])
    data_var = Variable("Data", is_independent=False, is_binned=False, units=y_unit)
    data_var.values = round_values(data_vals, ndigits)
    data_stat = Uncertainty("stat", is_symmetric=True)
    data_stat.values = round_uncertainties(data_errs, ndigits)
    data_var.add_uncertainty(data_stat)
    table.add_variable(data_var)

    # Total prediction
    hsum = _empty_like(template, "hTotalPrediction")
    for ic in SHOWN_COMPONENTS_SSWW:
        if ic in hist and hist[ic].GetSumOfWeights() != 0:
            hsum.Add(hist[ic])

    if htotal:
        for i in range(1, hsum.GetNbinsX() + 1):
            hsum.SetBinError(i, float(htotal.GetBinError(i)))

    _, total_vals, total_errs = _hist_to_edges_values_errors(hsum)
    total_var = Variable("Total prediction", is_independent=False, is_binned=False, units=y_unit)
    total_var.values = round_values(total_vals, ndigits)
    total_unc = Uncertainty("total", is_symmetric=True)
    total_unc.values = round_uncertainties(total_errs, ndigits)
    total_var.add_uncertainty(total_unc)
    table.add_variable(total_var)

    # MC components: central values only
    for ic in SHOWN_COMPONENTS_SSWW:
        if ic not in hist:
            continue
        h = hist[ic]
        if h.GetSumOfWeights() == 0:
            continue

        _, vals, _ = _hist_to_edges_values_errors(h)
        var = Variable(PLOT_LABELS_SSWW[ic], is_independent=False, is_binned=False, units=y_unit)
        var.values = round_values(vals, ndigits)
        table.add_variable(var)

    return table