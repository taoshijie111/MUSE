#!/usr/bin/env python
'''
Author: Ifsoul
Date: 2020-08-29 15:29:05
LastEditTime: 2022-03-01 17:15:53
LastEditors: Ifsoul
Description: Core funcions
'''
import math
import numpy as np
# from numpy import linalg as LA
from scipy import linalg as LA
from itertools import combinations
import sys
import time

from .data import Coor_Tol, Elements, Elem_Radius, hartree


def bond_chk(ele1, ele2, dis, Strict=True):
    '''Check whether a bond exist between ele1 and ele2 at distance dis'''
    if Strict:
        #(ra+rb)/rab>0.75(1-ln(1.1-1)/16)
        return (Elem_Radius[ele1 - 1] + Elem_Radius[ele2 - 1]) / dis > 0.857933676
    else:
        #(ra+rb)/rab>0.75(1-ln(1.333-1)/16)
        return (Elem_Radius[ele1 - 1] + Elem_Radius[ele2 - 1]) / dis > 0.801544349
        #(ra+rb)/rab>0.75(1-ln(2.5-1)/16)
        # return (Elem_Radius[ele1 - 1] + Elem_Radius[ele2 - 1]) / dis > 0.730993823

def atom_too_close_chk(ele1, ele2, dis):
    '''Check whether the distance between ele1 and ele2 is too small'''
    # return (Elem_Radius[ele1 - 1] + Elem_Radius[ele2 - 1]) / dis > 1.8
    return (Elem_Radius[ele1 - 1] + Elem_Radius[ele2 - 1]) / dis > 1.25

def distance(Ri, Rj):
    '''Distance between Ri and Rj'''
    return LA.norm(np.array(Ri) - np.array(Rj))


def distance2_vectors(Ris, Rjs):
    '''Square distance between vectors Ris and Rjs (both size: n x 3)'''
    r = Ris - Rjs
    return np.einsum('ij,ij->i', r, r)


def angle(Ri, Rj, Ro=np.zeros((3,))):
    '''Angle Ri-Ro-Rj'''
    R1 = np.array(Ri) - np.array(Ro)
    R2 = np.array(Rj) - np.array(Ro)
    return np.arccos(R1.dot(R2) / (LA.norm(R1) * LA.norm(R2)))


def atom_no(Names):
    '''Element names -> Element numbers'''
    if isinstance(Names, str):
        assert Names in Elements, "ERROR: Unkown atom type %s\n" % (Names)
        return Elements.index(Names) + 1
    else:
        No = []
        for ele in Names:
            assert ele in Elements, "ERROR: Unkown atom type %s\n" % (ele)
            No.append(Elements.index(ele) + 1)
        return np.array(No)


def atom_type(Nos):
    '''Element numbers -> Element names'''
    if isinstance(Nos, int):
        return Elements[Nos - 1]
    else:
        return np.array(Elements)[np.array(Nos).astype(int) - 1]


def num_map(cint):
    '''Convert a number (0~35) into char'''
    assert 36 >= cint >= 0, "ERROR: Illegal input (cint=%d)!\n" % cint
    if cint < 10:
        return chr(cint + 48)
    elif cint < 36:
        return chr(cint + 55)


def num2pos(nlist):
    '''Calculate the position of each item for the distance between items'''
    assert len(nlist)
    pos = [0]
    now = 0
    for n in nlist:
        now += n
        pos.append(now)
    return np.array(pos)


def int2str(num, digit=-1, base=36):
    '''Convert a number (base 10) into a string (digit: bit, base: base)'''
    assert base <= 36, "ERROR: Base(%d) is too large!\n" % base
    s = ''
    while True:
        x = num // base
        s += num_map(num % base)
        if x == 0:
            break
        num = x
    while len(s) < digit:
        s += '0'
    return s[::-1]


def str_sep(abc123):
    '''Cut a string like "abc123" into ["abc","123"]'''
    for c in abc123[::-1]:
        if not c.isdigit():
            break
    pos = abc123.rfind(c) + 1
    return abc123[:pos], abc123[pos:]


def sep_num_abc(string):
    '''Seperate numbers and letters in a string.
    Example: "ab45+-c123" -> ["ab", "45", "+-", "c", "123"]
    '''
    res = ['']
    current_type = 0 if string[0].isdigit() else 1 if string[0].isalpha() else -1
    for c in string:
        my_type = 0 if c.isdigit() else 1 if c.isalpha() else -1
        if my_type == current_type:
            res[-1] += c
        else:
            res.append(c)
            current_type = my_type
    return res


def hartree2nm(energy):
    '''Energy unit hartree -> wavelength nm'''
    return 1240.7011 / (energy * hartree)


def nm2hartree(wavelength):
    '''Wavelength nm -> Energy unit hartree'''
    return 1240.7011 / (wavelength * hartree)


def ev2nm(energy):
    '''Energy unit eV <-> wavelength nm'''
    return 1240.7011 / energy


def wavenum2nm(num):
    '''Wave number cm-1 <-> wavelength nm'''
    return 1e7 / num


def int2roman(num: int) -> str:
    a = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    b = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    res = ''
    for i, n in enumerate(a):
        while num >= a[i]:
            res += b[i]
            num -= a[i]
    return res


def roman2int(s: str) -> int:
    numbers = {
        'I': 1,
        'IV': 5,
        'V': 5,
        'IX': 10,
        'X': 10,
        'XL': 50,
        'L': 50,
        'XC': 100,
        'C': 100,
        'CD': 500,
        'D': 500,
        'CM': 1000,
        'M': 1000,
    }
    sum = 0
    n = 0
    while n <= len(s) - 1:
        if (numbers.get(s[n:n + 2])) != None and len(s[n:n + 2]) >= 2:
            sum = sum + numbers.get(s[n:n + 2]) - numbers[s[n]]
            n = n + 2
        else:
            sum = sum + numbers[s[n]]
            n = n + 1
    return sum


def find_all(Str, Tgt, Overlap=False):
    '''Find all Tgt in Str and return the positions.'''
    pos = []
    pst = 0
    dl = 1 if Overlap else len(Tgt)
    p = Str.find(Tgt, pst)
    while p >= 0:
        pos.append(p)
        pst = p + dl
        p = Str.find(Tgt, pst)
    return pos


def get_ext(Str, sep='.', ReturnName=False):
    '''Get the extension of a file'''
    p = Str.rfind(sep)
    if p < 0:
        p = len(Str)
    if ReturnName:
        return Str[:p], Str[p + 1:]
    else:
        return Str[p + 1:]


def force_ext(Fname, Ext) -> str:
    '''Check the extension of Fname. If not same as Ext, a new extension will be added.'''
    if Ext[0] != '.':
        Ext = '.' + Ext
    return Fname + (Ext if Fname[-len(Ext):] != Ext else '')


def linear_interp(Array_Like, Ratio):
    '''1D linear interpolation using the maximum and minimum of Array_Like'''
    return Ratio * np.max(Array_Like) + (1 - Ratio) * np.min(Array_Like)


def round_up_significant(number, digit=1):
    '''Round up a number with given number (default:1) of significant figures'''
    dgt = math.floor(math.log10(abs(number)))
    sft = 5 * 10**(dgt - digit)
    return round(number + math.copysign(sft, number), digit - 1 - dgt)


def sec2time(sec):
    '''Transform seconds into readable time format'''
    if isinstance(sec, str):
        sec = float(sec)
    assert isinstance(sec, int) or isinstance(sec, float), "ERROR: Input should be a number.\n"
    t = ' %gs' % (sec % 60)
    if sec >= 60:
        sec = sec // 60
        t = ' %dm' % (sec % 60) + t
        if sec >= 60:
            sec = sec // 60
            t = ' %dh' % (sec % 60) + t
            if sec >= 24:
                sec = sec // 24
                t = ' %dd' % (sec) + t
    return t


class timer(object):

    def __init__(self, name='', outfile=sys.stdout):
        self.name = name
        self.output = outfile

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print('%s time: %s' % (self.name, sec2time(self.end - self.start)), file=self.output)


def factor_pair(num):
    '''Factoring one number into two closest factors'''
    a = int(math.sqrt(num))
    while a > 1:
        if num % a == 0:
            return (a, num // a)
        a -= 1
    return (1, num)


def get_para(Arg_List, Num_Dict={'.': -1}, user_help_key=False, help=lambda: None):
    '''Get parameters from argument list'''
    if not user_help_key:
        Num_Dict.update({'h': 0, 'help': 0})
    para_pos = [i for i, p in enumerate(Arg_List) if p.startswith('-') and p[1:] in Num_Dict]
    para_Dict = {}
    for key in Num_Dict:
        if isinstance(Num_Dict[key], int):
            Num_Dict[key] = (Num_Dict[key], Num_Dict[key])
        elif Num_Dict[key][1] < 0:
            Num_Dict[key] = (Num_Dict[key][0], len(Arg_List))
    if '.' in Num_Dict:
        Num1 = para_pos[0] if para_pos else len(Arg_List)
        if Num_Dict['.'][0] >= 0 and not Num_Dict['.'][0] <= Num1 <= Num_Dict['.'][1]:
            print("ERROR: Unmatched number of parameters!")
            print("Expect %d%s parameter(s) but %d got.\n" % (Num_Dict['.'][0], '' if Num_Dict['.'][1] == Num_Dict['.'][0] else ' to %d' % Num_Dict['.'][1], Num1))
            help()
            sys.exit()
        else:
            para_Dict['.'] = Arg_List[:Num1]
    for k, i in enumerate(para_pos):
        key = Arg_List[i][1:]
        pos_st = i + 1
        pos_ed = para_pos[k + 1] if k + 1 < len(para_pos) else len(Arg_List)
        if Num_Dict[key][0] >= 0 and not Num_Dict[key][0] <= pos_ed - pos_st <= Num_Dict[key][1]:
            print("ERROR: Unmatched number of parameters for option -%s!" % key)
            print("Expect %d%s parameter(s) but %d got.\n" % (Num_Dict[key][0], '' if Num_Dict[key][1] == Num_Dict[key][0] else ' to %d' % Num_Dict[key][1], pos_ed - pos_st))
            help()
            sys.exit()
        para_Dict[key] = Arg_List[pos_st:pos_ed]
    if not user_help_key:
        if 'h' in para_Dict or 'help' in para_Dict:
            help()
            sys.exit()
    return para_Dict


def Str2List(Input_Str, range_mark=':~', sep_mark=','):
    '''Convert a string (list) into number list

    Examples:

    >>> a='12 3:4,6,7,14~17,9:11 20'
    >>> fs.Str2List(a)
    [12, 3, 4, 6, 7, 14, 15, 16, 17, 9, 10, 11, 20]

    >>> b='1!3;6;8 77@80'
    >>> fs.Str2List(b,'!@',';')
    [1, 2, 3, 6, 8, 77, 78, 79, 80]
    '''
    StrList = Input_Str.split() if isinstance(Input_Str, str) else Input_Str
    for m in sep_mark:
        StrList = [_ for s in StrList for _ in s.split(m)]
    Output_List = []
    for x in StrList:
        if x.isdigit():
            Output_List.append(int(x))
        else:
            for m in range_mark:
                if x.find(m) > 0:
                    S1, S2 = x.split(m)
                    break
            else:
                raise AssertionError('Unknown string format: %s' % x)
            assert S1.isdigit(), "ERROR: %s is not an integer\n" % S1
            assert S2.isdigit(), "ERROR: %s is not an integer\n" % S2
            Output_List.extend(range(int(S1), int(S2) + 1))
    return list(set(Output_List))


def Nums2Str(Nums, range_mark='-', sep_mark=','):
    """Convert a serial of number into one string

    Args:
        Nums (list): list of integers
        range_mark (str, optional): link mark for continuous numbers. Defaults to '-'.
        sep_mark (str, optional): seperation mark for non-continuous numbers. Defaults to ','.

    Returns:
        str: formated string
    """
    Nums = np.sort(Nums)
    d_Nums = [Nums[i + 1] - Nums[i] for i in range(len(Nums) - 1)]
    gap_pos = np.where(np.array(d_Nums) > 1)[0]
    st = [0] + (gap_pos + 1).tolist()
    ed = gap_pos.tolist() + [len(Nums) - 1]
    Output_List = []
    for i, j in zip(st, ed):
        if i == j:
            Output_List.append(str(Nums[i]))
        else:
            assert j > i
            Output_List.append(str(Nums[i]) + range_mark + str(Nums[j]))
    return sep_mark.join(Output_List)


def LongestCommonSublist(List1, List2):
    '''Search for the largest common sublist'''
    len1 = len(List1)
    len2 = len(List2)
    if len1 == 0 or len2 == 0:
        return 0, 0
    mtx = np.zeros([len1 + 1, len2 + 1], dtype=np.int)
    Lmax = 0
    for i in range(len1):
        for j in range(len2):
            if List1[i] == List2[j]:
                t = mtx[i][j] + 1
                mtx[i + 1][j + 1] = t
                if t > Lmax:
                    Lmax = t
    return Lmax, np.where(mtx == Lmax)


def trans_mtx(Source, Target):
    """Calculate the transforming matrix from Source to Target"""
    assert Source.shape == Target.shape, "ERROR: Different shapes between Source and Target!\n"
    assert Source.shape[1] == 3, "ERROR: Wrong lenth of coordinates!\n"
    Data = np.hstack((Source, np.ones((len(Source), 1))))
    # TransMatrix = LA.lstsq(Data, Target, rcond=None)[0].T
    TransMatrix = LA.lstsq(Data, Target)[0].T
    chk = (abs(TransMatrix[:3, :3]) < 1e-12)
    for i in range(0, 3):
        for j in range(0, 3):
            if all(chk[i, :]) and all(chk[:, j]):
                TransMatrix[i, j] = 1.0
    return TransMatrix


def Axis2mtx(Vector, Angle):
    """
    Calculate the transforming matrix.

    Parameters
    ----------
        Vector: (3,) array-like, rotating axis, should be a unit vector
        Angle:  float, rotating angle
    """
    # print('Axis:',Vector)
    tmp = np.array([[0, -Vector[2], Vector[1]], [Vector[2], 0, -Vector[0]], [-Vector[1], Vector[0], 0]])
    mtx = np.diag([1, 1, 1]) + tmp * math.sin(Angle) + np.dot(tmp, tmp) * (1 - math.cos(Angle))
    return mtx


def rotate_mtx(VctA, VctB):
    """Calculate the transforming matrix from vector VctA to vector VctB."""
    # print('VctA:',VctA)
    # print('VctB:',VctB)
    if np.allclose(VctA, VctB, atol=1e-8):
        return np.diag([1, 1, 1])
    Axis = np.cross(VctA, VctB)
    Axis = Axis / LA.norm(Axis)
    Angle = math.acos(np.dot(VctA, VctB) / (LA.norm(VctA) * LA.norm(VctB)))
    return Axis2mtx(Axis, Angle)


def cmp_atoms(AtomsA, AtomsB):
    """Coordinates AtomsA and AtomsB are all close or not."""
    assert AtomsA.shape == AtomsB.shape, "ERROR: Unmatched shapes of atoms!\n"
    if np.allclose(AtomsA, AtomsB, atol=Coor_Tol):
        return True
    for atom in AtomsA:
        Found = False
        for test in AtomsB:
            if np.allclose(atom, test, atol=Coor_Tol):
                Found = True
                break
        if not Found:
            return False
    return True


def co_line(VctA, VctB):
    """Two vectors are collinear or not. VctA and Vcts are ndarray."""
    assert VctA.shape == VctB.shape, "ERROR: Unmatched shapes of vectors!\n"
    cosA = np.dot(VctA, VctB) / (LA.norm(VctA) * LA.norm(VctB))
    # print("CoLine:",cosA)
    if 1 - abs(cosA) < 2e-2:
        return True
    else:
        return False


def not_co_line_all(Vcts):
    """Not all vectors are collinear or not. Vcts is (N,3) ndarray."""
    assert Vcts.shape[1] == 3, "ERROR: Wrong shapes of vectors!\n"
    NoList = list(combinations(range(len(Vcts)), 3))
    for i, j, k in NoList:
        if co_line(Vcts[j, :] - Vcts[i, :], Vcts[k, :] - Vcts[i, :]):
            continue
        else:
            return [i, j, k]
    return []


def co_plane(Vcts):
    """All vectors are coplanar or not. Vcts is (N,3) ndarray."""
    assert Vcts.shape[1] == 3, "ERROR: Wrong shapes of vectors!\n"
    NoList = list(combinations(range(len(Vcts)), 3))
    Vcts = Vcts / np.dot(LA.norm(Vcts, axis=1, keepdims=True), np.ones((1, 3)))
    MixTimes = np.array([Vcts[k, :].dot(np.cross(Vcts[i, :], Vcts[j, :])) for i, j, k in NoList])
    # print("MixTimes:",MixTimes)
    if (abs(MixTimes) < 2e-2).all():
        return True
    else:
        return False


def form_rot_mtx(Matrix, Half=False):
    """
    Check the rotating matrix to be made up of orthogonal basis.

    Parameters
    ----------
        Matrix: (3,3) ndarray, input rotating matrix
        Half:   bool. If True, only check all zero rows and columns; if False, check all rows to be orthogonal basis

    Returns
    ----------
        NewMtx: (3,3) ndarray, new rotating matrix
    """
    Matrix = Matrix[:3, :3]
    chk = (abs(Matrix) < 1e-6)
    for i in range(0, 3):
        for j in range(0, 3):
            if all(chk[i, :]) and all(chk[:, j]):
                Matrix[i, j] = 1.0
                chk[i, j] = False
    if Half:
        return Matrix
    # print('Mtx:\n',Matrix)
    # print('orth:\n',LA.orth(Matrix))
    Va, Vb, Vc = [np.array(x) for x in Matrix.tolist()]
    Va /= LA.norm(Va)
    Vb -= Vb.dot(Va) * Va
    Vb /= LA.norm(Vb)
    Vc -= Vc.dot(Va) * Va + Vc.dot(Vb) * Vb
    Vc /= LA.norm(Vc)
    return np.array([Va, Vb, Vc])
    # Matrix=np.array([Va,Vb,Vc])
    # print('New orth:\n',Matrix)


def sep_array(Array, Tolerance=Coor_Tol, Min=None, Max=None):
    """Divide input Array into individral regions"""
    if Min:
        Array = Array[Array >= Min]
    if Max:
        Array = Array[Array <= Max]
    Array = np.sort(Array)
    delta = Array[1:] - Array[:-1]
    idx_SepPos = np.where(delta > Tolerance)[0] + 1
    idx_start = np.insert(idx_SepPos, 0, 0)
    idx_end = np.append(idx_SepPos, len(Array))
    num = idx_end - idx_start
    sec_low = Array[idx_start] - Tolerance
    sec_high = Array[idx_end - 1] + Tolerance
    idx_overlap = np.where(sec_low[1:] < sec_high[:-1])[0]
    avg_overlap = 0.5 * (sec_low[idx_overlap + 1] + sec_high[idx_overlap])
    sec_low[idx_overlap + 1] = sec_high[idx_overlap] = avg_overlap
    pos = np.array([sec_low, sec_high]).T

    # if not Min:
    #     Min = np.min(Array)
    # if not Max:
    #     Max = np.max(Array)
    # bin_edge = np.arange(Min, Max + Tolerance, Tolerance)
    # hist, bin_edge = np.histogram(Array, bins=bin_edge)
    # pos1 = [i for i in range(len(hist)) if (i == 0 and hist[i]) or (i and hist[i] and not hist[i - 1])]
    # pos2 = [i for i in range(len(hist)) if (i == len(hist) - 1 and hist[i]) or (i < len(hist) - 1 and hist[i] and not hist[i + 1])]
    # pos = np.array([[bin_edge[i], bin_edge[j + 1]] for i, j in zip(pos1, pos2)])
    # num = np.array([np.sum(hist[i:j + 1]) for i, j in zip(pos1, pos2)])
    return num, pos


def unique(List_Like):
    """Find unique elements in List_Like and return as a list"""
    UnqList = []
    for x in List_Like:
        if x in UnqList:
            continue
        else:
            UnqList.append(x)
    # for i, x in enumerate(List_Like):
    #     for j in range(i):
    #         if List_Like[j] == x:
    #             break
    #     else:
    #         UnqList.append(x)
    return UnqList


def count_in_list(List_Like, HASHABLE=True):
    """Count elements in List_Like and return CountDict
    
    CountDict
    ---
        key:    Unique element (HASHABLE is True) or Index of unique element (HASHABLE is False)
        value:  Number of element
    """
    CountDict = {}
    if HASHABLE:
        for x in List_Like:
            CountDict[x] = CountDict.get(x, 0) + 1
    else:
        for i, x in enumerate(List_Like):
            for j in range(i):
                if List_Like[j] == x:
                    CountDict[j] += 1
                    break
            else:
                CountDict[i] = 1
    return CountDict


def count_dict_add(dict1, dict2):
    """Add values in two CountDicts"""
    return {k: (dict1.get(k, 0) + dict2.get(k, 0)) for k in dict1.keys() | dict2.keys()}


def count_dict_minus(dict1, dict2):
    """Minus values in two CountDicts"""
    return {k: (dict1.get(k, 0) - dict2.get(k, 0)) for k in dict1.keys() | dict2.keys()}


def find_same(List_like, myno):
    """Return the index of same element in List_like, -1 for not found"""
    p = List_like[myno]
    for j in range(myno):
        if p == List_like[j]:
            return j
    return -1


def find_same_all(List_like, ConvertFunc=lambda x: x):
    """Return the index of all same element in List_like, -1 for not found"""
    idx_same = (-np.ones_like(List_like)).astype(int)
    for i, x in enumerate(List_like):
        y = ConvertFunc(x)
        for j in range(i):
            if List_like[j] == y:
                idx_same[i] = j
                break
    return idx_same


def calc_pi(n, Type=1):
    """Calculate pi from the sum of series. n is the cutoff number"""
    if Type == 0:
        x = 0
        for i in range(1, n):
            x += 1 / (i * i)
        return math.sqrt(x * 6)
    elif Type == 1:
        a, b, t, x = (0, 1, 1, 1)
        for i in range(1, n):
            a += 1
            b += 2
            t *= a / b
            x += t
        return 2 * x


def best_pos(X, Y, BoxXRate, BoxYRate):
    """Find the best position of text avoiding cover curves"""
    XRescale = (np.array(X) - np.min(X)) / (np.max(X) - np.min(X))
    YRescale = (np.array(Y) - np.min(Y)) / (np.max(Y) - np.min(Y))
    GoodPoints = [[] for _ in range(6)]
    XBound = 1 - BoxXRate
    YBound = 1 - BoxYRate
    for x0 in np.arange(0, XBound, 0.01):
        idx = np.where((XRescale >= x0) & (XRescale <= (x0 + BoxXRate)))[0]
        YInBox = YRescale[idx]
        Ymax, Ymin = np.max(YInBox), np.min(YInBox)
        if x0 < XBound / 3:
            k = 0
        elif x0 < 2 * XBound / 3:
            k = 1
        else:
            k = 2
        if Ymax > YBound or Ymin < BoxYRate:
            continue
        else:
            if Ymin >= BoxYRate:
                for y0 in np.arange(0, Ymin - BoxYRate, 0.01):
                    GoodPoints[k].append((x0, y0))
            if Ymax <= YBound:
                for y0 in np.arange(Ymax, YBound - Ymax, 0.01):
                    GoodPoints[k + 1].append((x0, y0))
    NPoints = [len(pts) for pts in GoodPoints]
    NMax_Idx = np.argmax(NPoints)
    if not len(GoodPoints[NMax_Idx]):
        # print('WARNING: Fail to find the best location for text!')
        xpos, ypos = 0.6, 0.6
    else:
        xpos, ypos = np.mean(np.array(GoodPoints[NMax_Idx]), axis=0)
    return xpos, ypos


def fill_head_tail(XYMtx, Xst, Xed, FillValue=0):
    """Fill curve(s) by target value"""
    x = XYMtx[:, 0]
    Ny = XYMtx.shape[1] - 1
    dx = x[1] - x[0]
    NewMtx = XYMtx
    if x[0] > Xst + 2 * dx:
        xtmp = np.arange(Xst, x[0], dx).reshape((-1, 1))
        xy_head = np.hstack([xtmp, np.full((len(xtmp), Ny), FillValue)])
        NewMtx = np.vstack([xy_head, NewMtx])
    if x[-1] < Xed - 2 * dx:
        xtmp = np.arange(x[-1], Xed, dx).reshape((-1, 1))
        xy_tail = np.hstack([xtmp, np.full((len(xtmp), Ny), FillValue)])
        NewMtx = np.vstack([NewMtx, xy_tail])
    return NewMtx


def split_num(Number, N, Type='uniform'):
    """Divide a number into N parts"""
    LowType = Type.lower()
    if LowType == 'largefirst':
        LSmall = Number // N
        NLarge = Number % N
        SecEnd = 0
        OutList = [0]
        for i in range(N):
            if i < NLarge:
                SecEnd += LSmall + 1
            else:
                SecEnd += LSmall
            OutList.append(SecEnd)
        return OutList
    elif LowType == 'smallfirst':
        LSmall = Number // N
        NLarge = Number % N
        SecEnd = 0
        OutList = [0]
        for i in range(N):
            if i >= N - NLarge:
                SecEnd += LSmall + 1
            else:
                SecEnd += LSmall
            OutList.append(SecEnd)
        return OutList
    else:
        Lsec = Number / N
        return [round(i * Lsec) for i in range(N + 1)]


def common_values(lists):
    """Find all common values in input lists"""
    NList = len(lists)
    AllValues = []
    for l in lists:
        AllValues.extend(list(set(l)))
    return [k for k, v in count_in_list(AllValues).items() if v == NList]
