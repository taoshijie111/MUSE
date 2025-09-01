#!/usr/bin/env python
import numpy as np

from .data import Shells, ShellNum, Elements
from .core import atom_no, atom_type


def Basis2Dict(basis, elems=[]):
    blist = basis.split(',')
    my_elems = np.unique(elems) if len(elems) else np.arange(len(Elements))
    dict_out = {}
    for b in blist:
        elist = b.split(':')
        if len(elist) == 1:
            dict_out.update({e: elist[0] for e in my_elems})
        else:
            dict_out.update({e: elist[-1] for e in atom_no(elist[:-1]) if e in my_elems})
    return dict_out


def basis_compress(basis, elems):
    if basis.find(',') == -1:
        return basis.split(':')[-1]
    else:
        bdict = Basis2Dict(basis, elems)
        edict = {}
        for k, v in bdict.items():
            if v in edict:
                edict[v].append(atom_type(k))
            else:
                edict[v] = [atom_type(k)]
        if len(edict) == 1:
            return list(edict)[0]
        else:
            return ','.join([':'.join(v + [k]) for k, v in edict.items()])


def Ele2BasisName(element_list, basis='6-31g*'):
    bdict = Basis2Dict(basis, element_list)
    # assert basis in Shells[0][1:], "ERROR: Unknown basis: %s\n"%basis
    # BasisPos=Shells[0].index(basis)
    EleList = np.unique(element_list).tolist()
    SNum = [Shells[i][0] for i in range(1, len(Shells))]
    ElePos = []
    EleBasis = []
    for ele in EleList[:]:
        assert ele in bdict, 'ERROR: Unknown element %s in basisset %s!\n' % (atom_type(ele), basis)
        assert bdict[ele] in Shells[0][1:], "ERROR: Unknown basis: %s\n" % bdict[ele]
        BasisPos = Shells[0].index(bdict[ele])
        for i, n in enumerate(SNum):
            ele -= n
            if ele <= 0:
                ElePos.append(i + 1)
                Btmp = list(Shells[i + 1][BasisPos])
                EleBasis.append([b for b in Btmp for _ in range(ShellNum[b])])
                break
    for k, Blist in enumerate(EleBasis):
        Ltmp = np.array(Blist)
        NumCount = [np.sum(Ltmp[:i] == Blist[i]) for i in range(len(Blist))]
        EleBasis[k] = [a + str(b) for a, b in zip(Blist, NumCount)]
    # print(EleBasis)
    BasisOut = []
    for i, ele in enumerate(element_list):
        No = EleList.index(ele)
        suffix = '_%s%d' % (atom_type(ele), i)
        BasisOut.extend([a + suffix for a in EleBasis[No]])
    return BasisOut


def get_basisnum(basisset, eleno):
    if basisset in Shells[0][1:]:
        BasisPos = Shells[0].index(basisset)
    else:
        EleBasis = {}
        for sep in basisset.split(','):
            stmp = sep.split(':')
            for ele in stmp[:-1]:
                EleBasis[atom_no(ele)] = stmp[-1]
                assert stmp[-1] in Shells[0][1:], "ERROR: Unknown basis: %s\n" % stmp[-1]
        assert eleno in EleBasis, 'ERROR: Unknown element %s in basisset %s!\n' % (atom_type(eleno), basisset)
        BasisPos = Shells[0].index(EleBasis[eleno])
    SNum = [Shells[i][0] for i in range(1, len(Shells))]
    for i, n in enumerate(SNum):
        eleno -= n
        if eleno <= 0:
            MyShell = list(Shells[i + 1][BasisPos])
            return np.sum(np.array([ShellNum[b] for b in MyShell]))


def Ele2BasisPos(element_list, basisset='6-31g*'):
    current = 0
    result = []
    BasisPos = [0]
    for ele in element_list:
        num_basis = get_basisnum(basisset, ele)
        result.append(np.arange(current, current + num_basis))
        current += num_basis
        BasisPos.append(current)
    return np.array(result, dtype=object), BasisPos
