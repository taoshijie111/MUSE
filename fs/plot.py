#!/usr/bin/env python
'''
Author: Ifsoul
Date: 2020-08-29 16:01:12
LastEditTime: 2022-03-10 22:01:44
LastEditors: Ifsoul
Description: Functions for plotting figures
'''
import numpy as np
import math
import os
import matplotlib.pyplot as plt

from .data import hartree


def set_plt():
    plt.switch_backend('agg')
    # plt.rcParams["font.family"] = 'Times New Roman'
    # plt.rcParams["font.family"] = 'Arial Unicode MS'
    if os.name == 'nt':
        plt.rcParams["font.family"] = 'Arial'
        if 'font.sans-serif' in plt.rcParams:
            # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + plt.rcParams['font.sans-serif']
            plt.rcParams['font.sans-serif'].append('Microsoft YaHei')
            # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False


def energy_axis(Range, GHalf=0.001, MultiplyHartree=False):
    """Values of the energy axis for DOS curve"""
    Emin, Emax = Range
    dE = min((Emax - Emin) / 1000, GHalf / 100)
    SpreadLength = 100 * GHalf
    X = np.arange(Emin - SpreadLength, Emax + SpreadLength, dE)
    idx_out = np.where((X >= Emin) & (X <= Emax))[0]
    if MultiplyHartree:
        X *= hartree
    return X[idx_out]


def level2curve(EnergyLevel, Range, GHalf=0.001, Weight=None, Norm=False, Lorentzian=True, MultiplyHartree=False, YOnly=False):
    """Convert energy levels into DOS curve"""
    Emin, Emax = Range
    dE = min((Emax - Emin) / 1000, GHalf / 100)
    SpreadLength = 100 * GHalf
    gaussX = np.arange(-SpreadLength, SpreadLength, dE)
    gaussN = len(gaussX)
    if Lorentzian:
        gaussY = GHalf / (2 * np.pi * (gaussX**2 + 0.25 * GHalf * GHalf))
    else:
        s_gauss = GHalf * GHalf / (8 * np.log(2))
        gaussY = np.exp(-(0.5 * gaussX**2 / s_gauss)) / np.sqrt(2 * np.pi * s_gauss)
    X = np.arange(Emin - SpreadLength, Emax + SpreadLength, dE)
    idx_out = np.where((X >= Emin) & (X <= Emax))[0]
    Y = np.zeros_like(X)

    if Weight is not None:
        assert len(Weight) == len(EnergyLevel), 'ERROR: Length of Weight and EnergyLevel not equal (%d != %d).\n' % (len(Weight), len(EnergyLevel))
    else:
        Weight = np.ones_like(EnergyLevel)
    idx = np.where((EnergyLevel >= Emin - SpreadLength) & (EnergyLevel <= Emax + SpreadLength))[0]
    ED = EnergyLevel[idx]
    pos_list = np.zeros((len(ED), 4))
    pos_list[:, 0] = ((Emin - ED) / dE + 0.5).astype(int)
    pos_list[:, 1] = ((Emax + 2 * SpreadLength - ED) / dE + 0.5).astype(int)
    pos_list[:, 2] = ((ED - Emin) / dE + 0.5).astype(int)
    pos_list[:, 3] = ((ED + 2 * SpreadLength - Emin) / dE + 0.5).astype(int)
    pos_list[pos_list[:, 0] < 0, 0] = 0
    pos_list[pos_list[:, 1] > gaussN, 1] = gaussN
    pos_list[pos_list[:, 2] < 0, 2] = 0
    pos_list[pos_list[:, 3] > len(X), 3] = len(X)
    pos_list[:, 3] += pos_list[:, 1] - pos_list[:, 0] - pos_list[:, 3] + pos_list[:, 2]
    for w, pos in zip(Weight[idx], pos_list.astype(int)):
        Y[pos[2]:pos[3]] += w * gaussY[pos[0]:pos[1]]
    if Norm:
        Y /= len(EnergyLevel)
    if YOnly:
        return Y[idx_out]
    if MultiplyHartree:
        X *= hartree
    return np.array([X, Y])[:, idx_out].T


def multilevel2curves(EnergyLevels, Range, GHalf=0.001, Weights=None, Norm=False, Lorentzian=True, MultiplyHartree=False, YOnly=False):
    """Convert multiple energy levels into DOS curves"""
    Emin, Emax = Range
    dE = min((Emax - Emin) / 1000, GHalf / 100)
    SpreadLength = 100 * GHalf
    gaussX = np.arange(-SpreadLength, SpreadLength, dE)
    gaussN = len(gaussX)
    if Lorentzian:
        gaussY = GHalf / (2 * np.pi * (gaussX**2 + 0.25 * GHalf * GHalf))
    else:
        s_gauss = GHalf * GHalf / (8 * np.log(2))
        gaussY = np.exp(-(0.5 * gaussX**2 / s_gauss)) / np.sqrt(2 * np.pi * s_gauss)
    X = np.arange(Emin - SpreadLength, Emax + SpreadLength, dE)
    idx_out = np.where((X >= Emin) & (X <= Emax))[0]

    NYs = len(EnergyLevels)
    Y = np.zeros((NYs, len(X)))
    if Weights is not None:
        assert len(Weights) == NYs, 'ERROR: Numbers of Weights and EnergyLevels not equal (%d != %d).\n' % (len(Weights), NYs)
    else:
        Weights = [np.ones_like(_) for _ in EnergyLevels]
    for i, (el, wt) in enumerate(zip(EnergyLevels, Weights)):
        assert len(wt) == len(el), 'ERROR: Length of Weight and EnergyLevel No. %d not equal (%d != %d).\n' % (i, len(wt), len(el))
        idx = np.where((el >= Emin - SpreadLength) & (el <= Emax + SpreadLength))[0]
        ED = el[idx]
        pos_list = np.zeros((len(ED), 4))
        pos_list[:, 0] = ((Emin - ED) / dE + 0.5).astype(int)
        pos_list[:, 1] = ((Emax + 2 * SpreadLength - ED) / dE + 0.5).astype(int)
        pos_list[:, 2] = ((ED - Emin) / dE + 0.5).astype(int)
        pos_list[:, 3] = ((ED + 2 * SpreadLength - Emin) / dE + 0.5).astype(int)
        pos_list[pos_list[:, 0] < 0, 0] = 0
        pos_list[pos_list[:, 1] > gaussN, 1] = gaussN
        pos_list[pos_list[:, 2] < 0, 2] = 0
        pos_list[pos_list[:, 3] > len(X), 3] = len(X)
        pos_list[:, 3] += pos_list[:, 1] - pos_list[:, 0] - pos_list[:, 3] + pos_list[:, 2]
        for w, pos in zip(wt[idx], pos_list.astype(int)):
            Y[i, pos[2]:pos[3]] += w * gaussY[pos[0]:pos[1]]
        if Norm:
            Y[i, :] /= len(el)
    if YOnly:
        return Y[:, idx_out]
    if MultiplyHartree:
        X *= hartree
    return np.vstack([X.reshape((1, -1)), Y])[:, idx_out].T


def fig_double_y(
        FileName,
        XY_List,
        Labels=None,
        Type='curve',
        FigSize=(21.6, 9.0),
        AxisTitle=('axis x', 'axis y', 'axis y2'),
        Title='',
        AxisSize=(0.05, 0.1, 0.85, 0.82),
        XLimit=[],
        XLine=None,
        YLine=None,
        XShow=True,
        YShow=True,
        TickSize=14,
        LabelSize=16,
        TitleSize=16,
        LegendLoc=0,
        LegendCol=1,
        LegendShow=True,
        PDF=False,
        Styles=[],
):
    """Plot double-Y figure and save as file"""
    set_plt()
    fig = plt.figure(figsize=FigSize)
    assert Type in ('curve', 'point')
    if isinstance(XY_List, np.ndarray):
        assert XY_List.shape[1] > 2
        RealList = []
        for i in range(1, XY_List.shape[1]):
            RealList.append(XY_List[:, [0, i]])
        XY_List = RealList
    NPlot = len(XY_List)
    assert NPlot == 2
    if Styles:
        assert len(Styles) == NPlot, 'Unmatched number of Styles!'
    else:
        Styles = [{} for _ in range(NPlot)]
    if Labels is not None:
        assert len(Labels) == NPlot
        for i in range(NPlot):
            Styles[i]['label'] = Labels[i]
    ax1 = fig.add_axes(AxisSize)
    X = XY_List[0][:, 0]
    Y = XY_List[0][:, 1]
    if Type == 'curve':
        ax1.plot(X, Y, **Styles[0])
    elif Type == 'point':
        ax1.scatter(X, Y, **Styles[0])
    if XLimit:
        Xst, Xed = XLimit
    else:
        Xst = np.min(X)
        Xed = np.max(X)
    Yst, Yed = ax1.get_ylim()
    if XLine:
        for y in XLine:
            ax1.plot([Xst, Xed], [y, y], 'k--')
    if YLine:
        NLines = len(YLine)
        TextX = 0.003 * (Xed - Xst)
        TextY = [0.05 * i * Yed + (1 - 0.05 * i) * Yst for i in range(1, 6)] * math.ceil(NLines / 5)
        for i, x in enumerate(YLine):
            if isinstance(x, (int, float)):
                ax1.plot([x, x], [Yst, Yed], 'k--')
            else:
                x, t = x
                ax1.plot([x, x], [Yst, Yed], 'k--')
                plt.text(x + TextX, TextY[i], t, fontsize=TickSize)
    ax1.set_xlim(Xst, Xed)
    ax1.set_ylim(Yst, Yed)
    if Title:
        ax1.set_title(Title, fontsize=TitleSize)
    ax1.axes.xaxis.set_visible(XShow)
    if XShow:
        ax1.set_xlabel(AxisTitle[0], fontsize=LabelSize)
        ax1.tick_params(axis='x', length=TickSize)
        # for tick in ax1.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(TickSize)

    ax2 = ax1.twinx()
    X = XY_List[1][:, 0]
    Y = XY_List[1][:, 1]
    if Type == 'curve':
        ax2.plot(X, Y, **Styles[1])
    elif Type == 'point':
        ax2.scatter(X, Y, **Styles[1])
    ax1.axes.yaxis.set_visible(YShow)
    ax2.axes.yaxis.set_visible(YShow)
    if YShow:
        ax1.set_ylabel(AxisTitle[1], fontsize=LabelSize)
        ax2.set_ylabel(AxisTitle[2], fontsize=LabelSize)
        ax1.tick_params(axis='y', length=TickSize)
        ax2.tick_params(axis='y', length=TickSize)
        # for tick in ax1.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(TickSize)
        # for tick in ax2.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(TickSize)
    if LegendShow and Labels is not None:
        ax1.legend(loc=LegendLoc, fontsize=TickSize, ncol=LegendCol)
        ax2.legend(loc=LegendLoc, fontsize=TickSize, ncol=LegendCol)

    # ax1.plot(x, y1,'r',label="right");
    # ax1.legend(loc=1)
    # ax1.set_ylabel('Y values for exp(-x)');
    # ax2 = ax1.twinx() # this is the important function
    # ax2.plot(x, y2, 'g',label = "left")
    # ax2.legend(loc=2)
    # ax2.set_xlim([0, np.e]);
    # ax2.set_ylabel('Y values for ln(x)');
    # ax2.set_xlabel('Same X for both exp(-x) and ln(x)');

    plt.savefig(FileName, dpi=300)
    if PDF:
        plt.savefig(FileName[:FileName.rfind('.')] + '.pdf', dpi=300)


def fig_XYList(
        FileName,
        XY_List,
        Labels=None,
        Type='curve',
        FigSize=(14.4, 9.0),
        AxisTitle=('axis x', 'axis y'),
        Title='',
        AxisSize=(0.1, 0.1, 0.85, 0.82),
        Shift=0,
        XLine=None,
        YLine=None,
        XShow=True,
        YShow=True,
        TickSize=12,
        LabelSize=14,
        TitleSize=16,
        LegendSize=None,
        YBlank=0,
        LegendLoc=0,
        LegendCol=1,
        LegendShow=True,
        PDF=False,
        Styles=[],
):
    """Plot XY_List and save as figure file"""
    set_plt()
    fig = plt.figure(figsize=FigSize)
    assert Type in ('curve', 'point')
    NPlot = len(XY_List)
    if Styles:
        assert len(Styles) == NPlot, 'Unmatched number of Styles!'
    else:
        Styles = [{} for _ in range(NPlot)]
    if Labels is not None:
        assert len(Labels) == NPlot
        for i in range(NPlot):
            Styles[i]['label'] = Labels[i]
    ax = fig.add_axes(AxisSize)
    for i in range(NPlot):
        X = XY_List[i][:, 0]
        Y = XY_List[i][:, 1] + Shift * i
        if Type == 'curve':
            ax.plot(X, Y, **Styles[i])
        elif Type == 'point':
            ax.scatter(X, Y, **Styles[i])
        if not i:
            Xst = np.min(X)
            Xed = np.max(X)
        else:
            Xst = min(Xst, np.min(X))
            Xed = max(Xed, np.max(X))
    # Xst, Xed, Yst, Yed = plt.axis()
    # Yst, Yed = plt.axis()[2:]
    Yst, Yed = ax.get_ylim()
    Yed = max(Yed, Shift)
    Yed += YBlank * Shift
    if XLine:
        for y in XLine:
            ax.plot([Xst, Xed], [y, y], 'k--')
    if YLine:
        TextY = 0.1 * Yst + 0.9 * Yed
        for x in YLine:
            if isinstance(x, [int, float]):
                ax.plot([x, x], [Yst, Yed], 'k--')
            else:
                x, t = x
                ax.plot([x, x], [Yst, Yed], 'k--')
                plt.text(x, TextY, t)
    ax.set_xlim(Xst, Xed)
    ax.set_ylim(Yst, Yed)
    # plt.axis([Xst, Xed, Yst, Yed])
    if Title:
        ax.set_title(Title, fontsize=TitleSize)
    ax.axes.xaxis.set_visible(XShow)
    ax.axes.yaxis.set_visible(YShow)
    if XShow:
        ax.set_xlabel(AxisTitle[0], fontsize=LabelSize)
        ax.tick_params(axis='x', length=TickSize)
        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(TickSize)
    if YShow:
        ax.set_ylabel(AxisTitle[1], fontsize=LabelSize)
        ax.tick_params(axis='y', length=TickSize)
        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(TickSize)
    if LegendShow and Labels is not None:
        if LegendSize is None:
            LegendSize = TickSize
        ax.legend(loc=LegendLoc, fontsize=LegendSize, ncol=LegendCol)
    plt.savefig(FileName, dpi=300)
    if PDF and FileName[-4:] != '.pdf':
        plt.savefig(FileName[:FileName.rfind('.')] + '.pdf', dpi=300)
    print('Successfully saved figure: ', FileName)


def fig_XYMtx(
        FileName,
        XYMtx,
        Labels=None,
        Type='curve',
        FigSize=(14.4, 9.0),
        AxisTitle=('axis x', 'axis y'),
        Title='',
        AxisSize=(0.1, 0.1, 0.85, 0.82),
        XLine=None,
        YLine=None,
        XShow=True,
        YShow=True,
        BoxShow=True,
        Text=[],
        TickSize=12,
        LabelSize=14,
        TitleSize=16,
        LegendSize=None,
        PDF=False,
        XLimit=[],
        YLimit=[],
        Styles=[],
):
    """Plot XYMtx and save as figure file"""
    set_plt()
    fig = plt.figure(figsize=FigSize)
    ax = fig.add_axes(AxisSize)
    X = XYMtx[:, 0]
    Ys = XYMtx[:, 1:]
    NPlot = Ys.shape[1]
    assert Type in ('curve', 'point')
    if Styles:
        assert len(Styles) == NPlot, 'Unmatched number of Styles!'
    else:
        Styles = [{} for _ in range(NPlot)]
    if Labels is not None:
        assert len(Labels) == NPlot, 'Unmatched number of Labels!'
        for i in range(NPlot):
            Styles[i]['label'] = Labels[i]
    plot_handles = []
    if Type == 'curve':
        for i in range(NPlot):
            plot_handles.append(ax.plot(X, Ys[:, i], **Styles[i]))
        # Xst, Xed = (X[0], X[-1])
        Xst, Xed = XLimit if XLimit else (X[0], X[-1])
    elif Type == 'point':
        for i in range(NPlot):
            plot_handles.append(ax.scatter(X, Ys[:, i], **Styles[i]))
        Xst, Xed = XLimit if XLimit else (np.min(X), np.max(X))
        # Xst = np.min(X)
        # Xed = np.max(X)

    Yst, Yed = YLimit if YLimit else ax.get_ylim()
    # Yst, Yed = ax.get_ylim()
    if XLine:
        for y in XLine:
            ax.plot([Xst, Xed], [y, y], 'k--')
    if YLine:
        for x in YLine:
            ax.plot([x, x], [Yst, Yed], 'k--')
    ax.set_xlim(Xst, Xed)
    ax.set_ylim(Yst, Yed)
    if Title:
        plt.title(Title, fontsize=TitleSize)
    if Text:
        if isinstance(Text, list) and len(Text) > 1:
            for t in Text:
                plt.text(*t)
        else:
            plt.text(*Text)
    if XShow:
        plt.xlabel(AxisTitle[0], fontsize=LabelSize)
        plt.xticks(fontsize=TickSize)
    else:
        plt.xticks([])
    if YShow:
        plt.ylabel(AxisTitle[1], fontsize=LabelSize)
        plt.yticks(fontsize=TickSize)
    else:
        plt.yticks([])
    if Labels is not None:
        if LegendSize is None:
            LegendSize = TickSize
        plt.legend(fontsize=LegendSize)
    if not BoxShow:
        plt.axis('off')
    plt.savefig(FileName, dpi=300)
    if PDF and FileName[-4:] != '.pdf':
        plt.savefig(FileName[:FileName.rfind('.')] + '.pdf', dpi=300)
    print('Successfully saved figure: ', FileName)


def fig_dos(
        FileName,
        XYMtx,
        FigSize=(14.4, 9.0),
        AxisTitle=('Energy (eV)', 'Density of States'),
        *args,
        **kwargs,
):
    """Plot XYMtx and save as figure file"""
    fig_XYMtx(FileName, XYMtx, Type='curve', FigSize=FigSize, AxisTitle=AxisTitle, *args, **kwargs)


def fig_curves(FileName, XY_List, *args, **kwargs):
    """Plot XY_List and save as figure file"""
    fig_XYList(FileName, XY_List, Type='curve', *args, **kwargs)


def fig_dos_energylevel(
        FileName,
        Evals,
        ERange,
        Labels=None,
        FigSize=(14.4, 9.0),
        AxisTitle=('Energy (eV)', 'Density of States'),
        AxisSize=(0.1, 0.1, 0.85, 0.82),
        Title='',
        TickSize=12,
        LabelSize=14,
        TitleSize=16,
        LegendSize=None,
        XShow=True,
        YShow=True,
        BoxShow=True,
        XLimit=[],
        YLimit=[],
        PLOTEL=True,
        Styles=[],
        LevelKwargs={},
        PDF=False,
):
    """Plot and save DOS and energy-level graph of Evals in target ERange"""
    if isinstance(Evals, np.ndarray):
        Evals = [Evals]
    elif not isinstance(Evals, list):
        raise TypeError('Evals should be a (list of) 1-dim ndarray!')
    NPlot = len(Evals)
    assert NPlot > 0 and Evals[0].ndim == 1, 'ERROR: Evals should be a (list of) 1-dim ndarray!'
    XY = multilevel2curves(Evals, ERange, **LevelKwargs)
    X = XY[:, 0]
    Y = XY[:, 1:]
    set_plt()
    fig = plt.figure(figsize=FigSize)
    ax = fig.add_axes(AxisSize)
    if Styles:
        if len(Styles) != NPlot:
            raise RuntimeError('Unmatched number of Styles! (%d != %d)' % (len(Styles), NPlot))
    else:
        Styles = [{} for _ in range(NPlot)]
    if Labels is not None:
        assert len(Labels) == NPlot, 'Unmatched number of Labels!'
        for i in range(NPlot):
            Styles[i]['label'] = Labels[i]
    for i in range(NPlot):
        ax.plot(X, Y[:, i], **Styles[i])
    Xst, Xed = XLimit if XLimit else (X[0], X[-1])
    Yst, Yed = YLimit if YLimit else ax.get_ylim()
    if PLOTEL:
        cmap = plt.get_cmap("tab10")
        for no, myeval in enumerate(Evals):
            mycolor = cmap(no)
            for el in myeval:
                if ERange[0] <= el <= ERange[1]:
                    x = el * hartree
                    ax.plot([x, x], [0, 0.1 * Yed], color=mycolor, linestyle='-')
    ax.set_xlim(Xst, Xed)
    ax.set_ylim(Yst, Yed)
    if Title:
        plt.title(Title, fontsize=TitleSize)
    ax.axes.xaxis.set_visible(XShow)
    ax.axes.yaxis.set_visible(YShow)
    if XShow:
        plt.xlabel(AxisTitle[0], fontsize=LabelSize)
        plt.xticks(fontsize=TickSize)
    if YShow:
        plt.ylabel(AxisTitle[1], fontsize=LabelSize)
        plt.yticks(fontsize=TickSize)
    if Labels is not None:
        if LegendSize is None:
            LegendSize = TickSize
        plt.legend(fontsize=LegendSize)
    # if FileName[-4:] == '.pdf' or PDF:
    #     plt.rcParams['pdf.fonttype'] = 42
    #     plt.rcParams['font.family'] = 'Calibri'
    if not BoxShow:
        plt.axis('off')
    plt.savefig(FileName, dpi=300)
    if PDF and FileName[-4:] != '.pdf':
        plt.savefig(FileName[:FileName.rfind('.')] + '.pdf', dpi=300)
    print('Successfully saved figure: ', FileName)


def fig_energylevel(
        FileName,
        Ys,
        Labels,
        FigSize=(14.4, 9.0),
        AxisTitle=('Energy (eV)', 'Density of States'),
        Title='',
        TickSize=12,
        LabelSize=14,
        TitleSize=16,
        LevelSize=80,
):
    """Plot energy-level graph of Ys and save as figure file"""
    set_plt()
    plt.figure(figsize=FigSize)
    assert len(Labels) == Ys.shape[1]
    for i in range(Ys.shape[1]):
        plt.plot((i + 1) * np.ones_like(Ys[:, i]), Ys[:, i], '_', markersize=LevelSize)
    if Title:
        plt.title(Title, fontsize=TitleSize)
    plt.xlabel(AxisTitle[0], fontsize=LabelSize)
    plt.ylabel(AxisTitle[1], fontsize=LabelSize)
    plt.xlim((0, Ys.shape[1] + 1))
    plt.xticks(np.arange(Ys.shape[1]) + 1, Labels, fontsize=LabelSize)
    plt.yticks(fontsize=TickSize)
    plt.legend(fontsize=TickSize)
    plt.savefig(FileName, dpi=300)
    print('Successfully saved figure: ', FileName)


def fig_hist(
        FileName,
        Data,
        FigSize=(14.4, 9.0),
        AxisTitle=('Value', 'Probability density'),
        Title='Distribution of values',
        PREC=1e-8,
        Log=True,
        Density=True,
        TickSize=12,
        LabelSize=14,
        TitleSize=16,
        LevelSize=80,
):
    """Plot hist graph of Data and save as figure file"""
    set_plt()
    plt.figure(figsize=FigSize)
    Data = abs(Data)
    Data[Data < PREC] = PREC
    if Log:
        Data = np.log10(Data)
    cnts, bins = np.histogram(Data, bins=500, density=Density)
    bins = (bins[1:] + bins[:-1]) / 2
    plt.plot(bins, cnts)
    if Title:
        plt.title(Title, fontsize=TitleSize)
    plt.xlabel(AxisTitle[0], fontsize=LabelSize)
    plt.ylabel(AxisTitle[1], fontsize=LabelSize)
    if Log:
        TickList = np.arange(int(np.log10(PREC)), np.max(Data) + 1, 1)
        plt.xticks(TickList, ['$10^{%d}$' % x for x in TickList], fontsize=TickSize)
    else:
        plt.xticks(fontsize=TickSize)
    plt.yticks(fontsize=TickSize)
    plt.savefig(FileName, dpi=300)
    print('Successfully saved figure: ', FileName)


def fig_mtx(
        FileName,
        Data,
        FigSize=(14.4, 9.0),
        AxisTitle=('No. of Base Function', 'No. of Base Function'),
        Title='',
        CMap='coolwarm',
        Prec=1e-8,
        TickSize=12,
        LabelSize=14,
        TitleSize=16,
        LevelSize=80,
        LOG10=False,
        XTickKwargs={},
        YTickKwargs={},
):
    set_plt()
    plt.figure(figsize=FigSize)
    if LOG10:
        plt.imshow(np.log10(Data), cmap=CMap)
        TickList = np.arange(np.log10(Prec), np.log10(np.max(Data)) + 1, 1)
    else:
        Nmax = max(Data.shape)
        plt.imshow(Data, cmap=CMap, extent=(0, Nmax - 1, 0, Nmax - 1))
    if Title:
        plt.title(Title, fontsize=TitleSize)
    plt.xlabel(AxisTitle[0], fontsize=LabelSize)
    plt.ylabel(AxisTitle[1], fontsize=LabelSize)
    plt.xticks(fontsize=TickSize, **XTickKwargs)
    plt.yticks(fontsize=TickSize, **YTickKwargs)
    cbar = plt.colorbar()
    if LOG10:
        cbar.set_ticks(TickList)
        cbar.set_ticklabels(['$10^{%d}$' % x for x in TickList])
    cbar.ax.tick_params(labelsize=TickSize)
    plt.savefig(FileName, dpi=300)
    print('Successfully saved figure: ', FileName)


def fig_points(
        FileName,
        Data,
        FigSize=(9.6, 7.2),
        AxisSize=(0.14, 0.12, 0.83, 0.83),
        AxisTitle=('X', 'Y'),
        Title='',
        TickSize=18,
        LabelSize=24,
        TitleSize=16,
        XLim=[],
        YLim=[],
        Text=[],
        XTickKwargs={},
        YTickKwargs={},
        ScatterKwargs={},
):
    set_plt()
    fig = plt.figure(figsize=FigSize)
    ax = fig.add_axes(AxisSize)
    plt.scatter(Data[:, 0], Data[:, 1], **ScatterKwargs)
    if Title:
        plt.title(Title, fontsize=TitleSize)
    plt.xlabel(AxisTitle[0], fontsize=LabelSize)
    plt.ylabel(AxisTitle[1], fontsize=LabelSize)
    plt.xticks(fontsize=TickSize, **XTickKwargs)
    plt.yticks(fontsize=TickSize, **YTickKwargs)
    x = Data[:, 0]
    y = Data[:, 1]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_mean = np.mean(x * y)
    xx_mean = np.mean(x * x)

    m = (x_mean * y_mean - xy_mean) / (x_mean**2 - xx_mean)
    b = y_mean - m * x_mean
    plt.plot(x, m * x + b, 'r-')
    Xst, Xed = XLim if XLim else (np.min(x), np.max(x))
    Yst, Yed = YLim if YLim else (np.min(y), np.max(y))
    ax.set_xlim(Xst, Xed)
    ax.set_ylim(Yst, Yed)
    if Text:
        if isinstance(Text, list) and len(Text) > 1:
            for t in Text:
                plt.text(*t)
        else:
            plt.text(*Text)
    # if FileName[-4:] == '.pdf':
    #     plt.rcParams['pdf.fonttype'] = 42
    #     plt.rcParams['font.family'] = 'Calibri'
    plt.savefig(FileName, dpi=300)
    print('Successfully saved figure: ', FileName)


def fig_points_hist(
        FileName,
        Data,
        FigSize=(9.6, 7.2),
        AxisSize=(0.14, 0.12, 0.83, 0.83),
        AxisTitle=('X', 'Y'),
        Title='',
        TickSize=18,
        LabelSize=24,
        TitleSize=16,
        XLim=[],
        YLim=[],
        Text=[],
        XTickKwargs={},
        YTickKwargs={},
        ScatterKwargs={},
):
    set_plt()
    fig = plt.figure(figsize=FigSize)
    # ax = fig.add_axes(AxisSize)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=AxisSize[0],
        right=AxisSize[0] + AxisSize[2],
        bottom=AxisSize[1],
        top=AxisSize[1] + AxisSize[3],
        wspace=0.05,
        hspace=0.05,
    )
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(Data[:, 0], Data[:, 1], **ScatterKwargs)
    if Title:
        ax.title(Title, fontsize=TitleSize)
    ax.set_xlabel(AxisTitle[0], fontsize=LabelSize)
    ax.tick_params(axis='x', length=TickSize, **XTickKwargs)
    ax.set_ylabel(AxisTitle[1], fontsize=LabelSize)
    ax.tick_params(axis='y', length=TickSize, **YTickKwargs)
    x = Data[:, 0]
    y = Data[:, 1]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_mean = np.mean(x * y)
    xx_mean = np.mean(x * x)

    m = (x_mean * y_mean - xy_mean) / (x_mean**2 - xx_mean)
    b = y_mean - m * x_mean
    ax.plot(x, m * x + b, 'r-')
    Xst, Xed = XLim if XLim else (np.min(x), np.max(x))
    Yst, Yed = YLim if YLim else (np.min(y), np.max(y))
    ax.set_xlim(Xst, Xed)
    ax.set_ylim(Yst, Yed)
    if Text:
        if isinstance(Text, list) and len(Text) > 1:
            for t in Text:
                ax.text(*t)
        else:
            ax.text(*Text)

    bin_num = 101
    bins = np.linspace(Xst, Xed, bin_num)
    ax_histx.hist(x, bins=bins)
    bins = np.linspace(Yst, Yed, bin_num)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    plt.savefig(FileName, dpi=300)
    print('Successfully saved figure: ', FileName)


def fig_shadow(
        FileName,
        XYMtx,
        udinfo=None,
        Labels=None,
        FigSize=(9.6, 7.2),
        AxisSize=(0.14, 0.12, 0.83, 0.83),
        AxisTitle=('Energy (eV)', 'Density of States'),
        Title='',
        TickSize=18,
        LabelSize=24,
        TitleSize=16,
        LegendSize=18,
        XLim=[],
        YLim=[],
        Text=[],
        Styles=[],
):
    set_plt()
    fig = plt.figure(figsize=FigSize)
    ax = fig.add_axes(AxisSize)
    X = XYMtx[:, 0]
    Ys = XYMtx[:, 1:]
    NPlot = Ys.shape[1]
    if Styles:
        assert len(Styles) == NPlot, 'Unmatched number of Styles!'
    else:
        Styles = [{} for _ in range(NPlot)]
    if Labels is not None:
        assert len(Labels) == NPlot, 'Unmatched number of Labels!'
        for i in range(NPlot):
            Styles[i]['label'] = Labels[i]
    for i in range(NPlot):
        ax.plot(X, Ys[:, i], **Styles[i])
    Xst, Xed = XLim if XLim else (X[0], X[-1])
    Yst, Yed = YLim if YLim else ax.get_ylim()
    plt.title(Title, fontsize=TitleSize)
    if udinfo is not None:
        for ud, er in udinfo:
            if ud in 'UD':
                ShadowColor = 'lightpink' if ud == 'U' else 'lightblue'
                plt.fill_between(er, Yst, Yed, color=ShadowColor, alpha=0.5)
                ax.plot([er[0], er[0]], [Yst, Yed], 'k--')
                ax.plot([er[1], er[1]], [Yst, Yed], 'k--')
            elif ud == 'H':
                ax.plot([er[0], er[0]], [Yst, Yed], 'k--')
    ax.set_xlim(Xst, Xed)
    ax.set_ylim(Yst, Yed)
    if NPlot > 1:
        plt.legend(fontsize=LegendSize)
    if Text:
        if isinstance(Text, list) and len(Text) > 1:
            for t in Text:
                plt.text(*t)
        else:
            plt.text(*Text)
    plt.xlabel(AxisTitle[0], fontsize=LabelSize)
    plt.ylabel(AxisTitle[1], fontsize=LabelSize)
    plt.xticks(fontsize=TickSize)
    plt.yticks(fontsize=TickSize)
    plt.savefig(FileName, dpi=300)
    plt.close()
    print('Successfully saved figure: ', FileName)
