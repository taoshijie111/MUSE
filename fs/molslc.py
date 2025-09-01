#!/usr/bin/env python
'''
Author: Ifsoul
Date: 2020-08-29 16:12:32
LastEditTime: 2021-04-01 20:33:43
LastEditors: Ifsoul
Description: Functions for molecule slices
'''
import numpy as np
import multiprocessing as mp
from scipy import linalg as LA
# from numpy import linalg as LA
import os
import copy

from .data import Coor_Tol, Basis_Dict
from .core import atom_no, atom_type, int2str, str_sep, sep_array, co_plane, not_co_line_all, co_line, cmp_atoms, form_rot_mtx
from .basis import basis_compress
from .parallel import Job
from .fileIO import read_file


def test_prop(Prop, Coors, NoList):
    '''Whether the input coordinates match the given property or not'''
    for j, k in NoList:
        Good = (np.abs(Prop[j] - np.sqrt(np.sum((Coors[j, :] - Coors[k, :])**2))) < 5 * Coor_Tol)
        # Good=(np.abs(Prop[j]-np.sqrt(np.sum(np.mean(Coors[[j,k],:],axis=0)**2))) < Coor_Tol)
        if not Good:
            return False
    return True


def get_sets(DisA, DisB, CoorA, CoorB):
    '''Get the atom pairs depending on distance'''
    Num, Sec = sep_array(DisA, Min=np.min([DisA, DisB]), Max=np.max([DisA, DisB]))
    SortPos = np.argsort(Num)
    NSet = min(len(Num), 10)
    if NSet <= 6:
        print('WARNING: Number of atoms (Nset=%d) too small! Result may be wrong!\n' % NSet)
    SetA = []
    SetB = []
    NAdd = 0
    if Num[SortPos[NSet - 1]] == 1 and len(Num) > NSet:
        tmp = np.where(Num > 2)[0]
        No = tmp[0] if len(tmp) else SortPos[-1]
        # SetA.append(np.random.choice(np.where((DisA>=Sec[No][0]) & (DisA<=Sec[No][1]))[0]))
        SetA.append(np.where((DisA >= Sec[No][0]) & (DisA <= Sec[No][1]))[0][0])
        SetB.append(np.where((DisB >= Sec[No][0]) & (DisB <= Sec[No][1]))[0])
        NAdd = 1
    # print('SetA:',SetA)
    # print('SetB:',SetB)
    for i in range(NSet - NAdd):
        # print(np.arange(NSet)-1)
        # for i in np.arange(NSet)-1:
        No = SortPos[i]
        SetA.append(np.where((DisA >= Sec[No][0]) & (DisA <= Sec[No][1]))[0][0])
        SetB.append(np.where((DisB >= Sec[No][0]) & (DisB <= Sec[No][1]))[0])
    # print('SetA:', SetA)
    # print('SetB:', SetB)
    # print('Co Plane:',co_plane(CoorA[SetA,:]))
    CoPln = False
    if co_plane(CoorA[SetA, :]):
        if NSet < len(Num):
            CoPln = True
            for i in range(NSet, len(Num)):
                No = SortPos[i]
                SetA[-1] = np.where((DisA >= Sec[No][0]) & (DisA <= Sec[No][1]))[0][0]
                # SetA[-1]=np.random.choice(np.where((DisA>=Sec[No][0]) & (DisA<=Sec[No][1]))[0])
                # print('SetA:',SetA)
                # print('Co Plane:',co_plane(CoorA[SetA,:]))
                if not co_plane(CoorA[SetA, :]):
                    SetB[-1] = np.where((DisB >= Sec[No][0]) & (DisB <= Sec[No][1]))[0]
                    CoPln = False
                    break
            if CoPln:
                No = SortPos[NSet - 1]
                SetA[-1] = np.where((DisA >= Sec[No][0]) & (DisA <= Sec[No][1]))[0][0]
                # SetA[-1]=np.random.choice(np.where((DisA>=Sec[No][0]) & (DisA<=Sec[No][1]))[0])
                if not not_co_line_all(CoorA[SetA, :]):
                    tmp = np.where(Num > 2)[0]
                    No = tmp[0] if len(tmp) else SortPos[-1]
                    for x in np.where((DisA >= Sec[No][0]) & (DisA <= Sec[No][1]))[0]:
                        SetA[-1] = x
                        if not_co_line_all(CoorA[SetA[-3:], :]):
                            break
                    SetB[-1] = np.where((DisB >= Sec[No][0]) & (DisB <= Sec[No][1]))[0]

    Ctmp = CoorA[SetA, :]
    NSet = len(Ctmp)
    dis_prop = [np.sqrt(np.sum((Ctmp[i, :] - Ctmp[i + 1, :])**2)) for i in range(NSet) if i + 1 < NSet]
    SetTmp = [[_] for _ in SetB[0]]
    for i, stest in enumerate(SetB[1:]):
        SetTmp2 = []
        for sold in SetTmp:
            C1 = CoorB[sold[-1], :]
            for snew in stest:
                C2 = CoorB[snew, :]
                if np.abs(dis_prop[i] - np.sqrt(np.sum((C1 - C2)**2))) < 2 * Coor_Tol:
                    SetTmp2.append(sold + [snew])
        SetTmp = SetTmp2
    Sets = [SetA] + SetTmp
    # print('Sets Len:', len(Sets))
    return CoPln, np.array(Sets)  # CoPln,pos


def get_sets_from_order(OrderA, OrderB, CoorA, CoorB, Debug=False):
    '''Get the atom pairs depending on distance order'''
    ListA = [[OrderA[0, 0]]]
    ListB = [[OrderB[0, 0]]]
    for i in range(1, len(OrderA)):
        if (OrderA[i, 1:4] == OrderA[i - 1, 1:4]).all():
            ListA[-1].append(OrderA[i, 0])
            ListB[-1].append(OrderB[i, 0])
        else:
            ListA.append([OrderA[i, 0]])
            ListB.append([OrderB[i, 0]])
    if Debug:
        print('ListA:', ListA)
        print('ListB:', ListB)
    NTotal = len(ListA)
    NSet = min(NTotal, 10)
    if NSet <= 6:
        print('WARNING: Number of atoms (Nset=%d) too small! Result may be wrong!\n' % NSet)
    SetA = [_[0] for _ in ListA[:NSet]]
    SetB = ListB[:NSet]
    if Debug:
        print('SetA:', SetA)
        print('SetB:', SetB)
    CoPln = co_plane(CoorA[SetA, :])
    if CoPln and NSet < NTotal:
        for i in range(NSet, NTotal):
            SetA[-1] = ListA[i][0]
            if not co_plane(CoorA[SetA, :]):
                SetB[-1] = ListB[i]
                CoPln = False
                break
        if CoPln:
            SetA[-1] = ListA[NSet - 1][0]
            if not not_co_line_all(CoorA[SetA, :]):
                No = NSet - 1
                for i in range(NSet, NTotal):
                    for x in ListA[i]:
                        SetA[-1] = x
                        if not_co_line_all(CoorA[SetA[-3:], :]):
                            No = i
                            break
                SetB[-1] = ListB[No]
        if Debug:
            print('SetA:', SetA)
            print('SetB:', SetB)

    Ctmp = CoorA[SetA, :]
    NSet = len(Ctmp)
    dis_prop = [np.sqrt(np.sum((Ctmp[i, :] - Ctmp[i + 1, :])**2)) for i in range(NSet) if i + 1 < NSet]
    SetTmp = [[_] for _ in SetB[0]]
    for i, stest in enumerate(SetB[1:]):
        SetTmp2 = []
        for sold in SetTmp:
            C1 = CoorB[sold[-1], :]
            for snew in stest:
                C2 = CoorB[snew, :]
                if np.abs(dis_prop[i] - np.sqrt(np.sum((C1 - C2)**2))) < 2 * Coor_Tol:
                    SetTmp2.append(sold + [snew])
        SetTmp = SetTmp2
    Sets = [SetA] + SetTmp
    if Debug:
        print('Sets Len:', len(Sets))
    return CoPln, np.array(Sets)  # CoPln,pos


class slice_name(object):
    '''Name of mol_slice'''

    def __init__(self, TYPE, LEN, ELEM, ENUM, SNO, RMARK, POS, RNO, BNO):
        self.Type = TYPE
        self.Len = LEN
        self.Elem = ELEM
        self.ENum = ENUM
        self.StrucNo = SNO
        self.RStat = RMARK
        self.Pos = POS
        self.RelaxNo = RNO
        self.BasisNo = BNO
        self.Str = '%s%d_%s%d_%s_%s_%s_%s' % (TYPE, LEN, atom_type(ELEM), ENUM, int2str(SNO, 6), (RMARK if RMARK != 'R' else RMARK + POS), int2str(RNO, 2), int2str(BNO, 2))

    def __eq__(self, other):
        assert isinstance(other, slice_name), "ERROR: Data type not match!\n"
        return (self.Str == other.Str)

    def __gt__(self, other):
        assert isinstance(other, slice_name), "ERROR: Data type not match!\n"
        if self.RStat == 'Z' and other.RStat == 'Z':
            return (self.Str > other.Str)
        elif self.RStat == 'Z':
            return True
        elif other.RStat == 'Z':
            return False
        else:
            if self.Type != other.Type:
                return (self.Type > other.Type)
            elif self.Len != other.Len:
                return (self.Len > other.Len)
            elif self.Elem != other.Elem:
                return (self.Elem > other.Elem)
            elif self.ENum != other.ENum:
                return (self.ENum > other.ENum)
            elif self.StrucNo != other.StrucNo:
                return (self.StrucNo > other.StrucNo)
            elif self.RStat != other.RStat:
                return (self.RStat > other.RStat)
            elif self.Pos != other.Pos:
                return (self.Pos > other.Pos)
            elif self.RelaxNo != other.RelaxNo:
                return (self.RelaxNo > other.RelaxNo)
            else:
                return (self.BasisNo > other.BasisNo)

    def __lt__(self, other):
        assert isinstance(other, slice_name), "ERROR: Data type not match!\n"
        if self.RStat == 'Z' and other.RStat == 'Z':
            return (self.Str < other.Str)
        elif self.RStat == 'Z':
            return False
        elif other.RStat == 'Z':
            return True
        else:
            if self.Type != other.Type:
                return (self.Type < other.Type)
            elif self.Len != other.Len:
                return (self.Len < other.Len)
            elif self.Elem != other.Elem:
                return (self.Elem < other.Elem)
            elif self.ENum != other.ENum:
                return (self.ENum < other.ENum)
            elif self.StrucNo != other.StrucNo:
                return (self.StrucNo < other.StrucNo)
            elif self.RStat != other.RStat:
                return (self.RStat < other.RStat)
            elif self.Pos != other.Pos:
                return (self.Pos < other.Pos)
            elif self.RelaxNo != other.RelaxNo:
                return (self.RelaxNo < other.RelaxNo)
            else:
                return (self.BasisNo < other.BasisNo)

    def __ne__(self, other):
        assert isinstance(other, slice_name), "ERROR: Data type not match!\n"
        return not (self == other)

    def __ge__(self, other):
        assert isinstance(other, slice_name), "ERROR: Data type not match!\n"
        return (self == other) or (self > other)

    def __le__(self, other):
        assert isinstance(other, slice_name), "ERROR: Data type not match!\n"
        return (self == other) or (self < other)

    def diff(self, other):
        assert isinstance(other, slice_name), "ERROR: Data type not match!\n"
        if self == other:
            return 0
        else:
            if self.Type != other.Type:
                return 1
            elif self.Len != other.Len:
                return 2
            elif self.Elem != other.Elem:
                return 3
            elif self.ENum != other.ENum:
                return 4
            elif self.StrucNo != other.StrucNo:
                return 5
            elif self.RStat != other.RStat:
                return 6
            elif self.Pos != other.Pos:
                return 7
            elif self.RelaxNo != other.RelaxNo:
                return 8
            else:
                return 9

    def get_str(self):
        '''String format'''
        self.Str = '%s%d_%s%d_%s_%s_%s_%s' % (self.Type, self.Len, atom_type(self.Elem), self.ENum, int2str(self.StrucNo, 6), (self.RStat if self.RStat != 'R' else self.RStat + self.Pos), int2str(self.RelaxNo, 2), int2str(self.BasisNo, 2))
        return self.Str

    def update(self, STR):
        '''Update by given string'''
        SList = STR.split('_')
        if len(SList) != 6:
            self.Str = STR
        else:
            stmp = str_sep(SList[0])
            self.Type = stmp[0]
            self.Len = int(stmp[1])
            stmp = str_sep(SList[1])
            self.Elem = atom_no(stmp[0])
            self.ENum = int(stmp[1])
            self.StrucNo = int(SList[2], 36)
            self.RStat = SList[3][0]
            if len(SList[3]) > 1:
                self.Pos = SList[3][1:]
            else:
                self.Pos = '0'
            self.RelaxNo = int(SList[4], 36)
            self.BasisNo = int(SList[5], 36)
            self.get_str()

    @classmethod
    def get_from_str(cls, STR, MSG=True):
        '''Get a new slice_name object from input string'''
        if cls.fmt_chk(STR, MSG):
            SList = STR.split('_')
            stmp = str_sep(SList[0])
            TYPE = stmp[0]
            LEN = int(stmp[1])
            stmp = str_sep(SList[1])
            ELEM = atom_no(stmp[0])
            ENUM = int(stmp[1])
            SNO = int(SList[2], 36)
            RMARK = SList[3][0]
            if len(SList[3]) > 1:
                POS = SList[3][1:]
            else:
                POS = '0'
            RNO = int(SList[4], 36)
            BNO = int(SList[5], 36)
            return cls(TYPE, LEN, ELEM, ENUM, SNO, RMARK, POS, RNO, BNO)
        else:
            new = cls('XX', 0, 1, 0, 0, 'Z', 'Z', 0, 0)
            new.Str = STR
            return new

    @staticmethod
    def cal_pos(No, NPart):
        '''Calculate the position value of part No in a molecule with NPart parts'''
        if NPart > 4:
            if No == 0:
                POS = 'E0'
            elif No == 1:
                POS = 'B0'
            elif No == NPart - 2:
                POS = 'B1'
            elif No == NPart - 1:
                POS = 'E1'
            else:
                POS = 'C'
        elif NPart > 2:
            if No == 0:
                POS = 'E0'
            elif No == NPart - 1:
                POS = 'E1'
            else:
                POS = ''
        else:
            POS = ''
        return POS

    @staticmethod
    def fmt_chk(STR, MSG=True):
        '''Check the format of string'''
        SList = STR.split('_')
        if len(SList) != 6:
            if MSG:
                print('Fail to format "%s": Wrong length!' % STR)
            return False
        else:
            for s in SList[1:]:
                if not s.isalnum():
                    if MSG:
                        print('Fail to format "%s": Wrong format of %s!' % (STR, s))
                    return False
            if SList[0][0].isalpha() and SList[0][-1].isdigit():
                stmp = str_sep(SList[0])
                # if not (stmp[0].isalpha() and stmp[1].isdigit()):
                if not stmp[1].isdigit():
                    if MSG:
                        print('Fail to format "%s": Wrong format of %s!' % (STR, SList[0]))
                    return False
            else:
                if MSG:
                    print('Fail to format "%s": Wrong format of %s!' % (STR, SList[0]))
                return False
            if SList[1][0].isalpha() and SList[1][-1].isdigit():
                stmp = str_sep(SList[1])
                if not (stmp[0].isalpha() and stmp[1].isdigit()):
                    if MSG:
                        print('Fail to format "%s": Wrong format of %s!' % (STR, SList[1]))
                    return False
            else:
                if MSG:
                    print('Fail to format "%s": Wrong format of %s!' % (STR, SList[1]))
                return False
            if SList[3][0] not in ['N', 'R']:
                if MSG:
                    print('Fail to format "%s": Wrong format of %s!' % (STR, SList[3]))
                return False
        return True


class mol_slice(object):
    '''Slice of molecule'''

    def __init__(self, NUM, PATH, NAME, ELEM, ENUM, STAT=0x01, BASIS='6-31g*'):
        self.AtomNum = NUM
        self.Path = PATH
        self.Proj = NAME
        self.Elem = ELEM
        self.ENum = ENUM
        self.Stat = STAT
        self.Basis = BASIS
        self.Data = []
        self.SortOrder = []
        self.Coor0Center = []
        self.DisInfo = []
        self.Key = slice_name.get_from_str(NAME, MSG=False)
        self.Mark = 0

    def __eq__(self, other):
        assert isinstance(other, mol_slice), "ERROR: Data type not match!\n"
        if (self.AtomNum == other.AtomNum) and (self.Basis == other.Basis) and (self.Key == other.Key) and (self.Path == other.Path):
            return (np.all(self.Elem == other.Elem) and np.all(self.ENum == other.ENum))
        else:
            return False

    def __gt__(self, other):
        assert isinstance(other, mol_slice), "ERROR: Data type not match!\n"
        if self.AtomNum == other.AtomNum:
            for i in range(min(len(self.Elem), len(other.Elem))):
                if self.Elem[i] == other.Elem[i]:
                    if self.ENum[i] != other.ENum[i]:
                        return (self.ENum[i] > other.ENum[i])
                else:
                    return (self.Elem[i] > other.Elem[i])
            if len(self.Elem) != len(other.Elem):
                return len(self.Elem) > len(other.Elem)
            elif self.Basis != other.Basis:
                return self.Basis > other.Basis
            elif self.Key != other.Key:
                return self.Key > other.Key
            else:
                return self.Path > other.Path
        else:
            return (self.AtomNum > other.AtomNum)

    def __lt__(self, other):
        assert isinstance(other, mol_slice), "ERROR: Data type not match!\n"
        if self.AtomNum == other.AtomNum:
            for i in range(min(len(self.Elem), len(other.Elem))):
                if self.Elem[i] == other.Elem[i]:
                    if self.ENum[i] != other.ENum[i]:
                        return (self.ENum[i] < other.ENum[i])
                else:
                    return (self.Elem[i] < other.Elem[i])
            if len(self.Elem) != len(other.Elem):
                return len(self.Elem) < len(other.Elem)
            elif self.Basis != other.Basis:
                return self.Basis < other.Basis
            elif self.Key != other.Key:
                return self.Key < other.Key
            else:
                return self.Path < other.Path
        else:
            return (self.AtomNum < other.AtomNum)

    def __ne__(self, other):
        assert isinstance(other, mol_slice), "ERROR: Data type not match!\n"
        return not (self == other)

    def __ge__(self, other):
        assert isinstance(other, mol_slice), "ERROR: Data type not match!\n"
        return (self == other) or (self > other)

    def __le__(self, other):
        assert isinstance(other, mol_slice), "ERROR: Data type not match!\n"
        return (self == other) or (self < other)

    def str(self):
        '''String format'''
        ElemInfo = ''
        EType = atom_type(self.Elem)
        for k in range(len(self.Elem)):
            ElemInfo = ElemInfo + ' %s %d' % (EType[k], self.ENum[k])
        return '%d\t%s\t%s\t%s\t%2X%s' % (self.AtomNum, self.Path, self.Proj, self.Basis, self.Stat, ElemInfo)

    def get_data(self, Update=False):
        '''Get atoms Data. Read from file if Update is True'''
        if Update or not len(self.Data):
            self.Data = read_file(self.Path + "atom_list_%s.csv" % self.Proj)[1]
        return self.Data

    @staticmethod
    def chk_files(PATH, NAME):
        '''Check the existence of files and return the STAT value'''
        STAT = 0x00
        if os.path.exists(PATH + "atom_list_%s.csv" % NAME):
            STAT = STAT | 0x01
        if os.path.exists(PATH + "H_%s.npz" % NAME):
            STAT = STAT | 0x02
        if os.path.exists(PATH + "S_%s.npz" % NAME):
            STAT = STAT | 0x04
        if os.path.exists(PATH + "MO_%s.npz" % NAME):
            STAT = STAT | 0x10
        if os.path.exists(PATH + "Evals_%s.npz" % NAME):
            STAT = STAT | 0x20
        return STAT

    @staticmethod
    def count_elements(ElemList):
        '''Count element types and numbers'''
        ElemList = np.array(ElemList)
        Elem = np.unique(ElemList)
        ENum = np.zeros_like(Elem)
        for i, x in enumerate(Elem):
            ENum[i] = np.sum(ElemList == x)
        snum = np.lexsort([-ENum, Elem])
        Elem = Elem[snum].astype(int)
        ENum = ENum[snum].astype(int)
        return Elem, ENum

    @staticmethod
    def read_basis_name(BasisFile, Elem):
        '''Read basis name from file'''
        with open(BasisFile, 'r') as F_in:
            Basis_Set = F_in.readline().split()[0]
        if Basis_Set in Basis_Dict:
            Basis = Basis_Dict[Basis_Set]
        else:
            Basis = basis_compress(Basis_Set, Elem)
        return Basis

    def update(self, Deep=False):
        '''Update information. Read atom and basis files if Deep is True'''
        PATH = self.Path
        NAME = self.Proj
        self.Stat = self.chk_files(PATH, NAME)
        if Deep and (self.Stat & 0x01):
            atoms = read_file(PATH + "atom_list_%s.csv" % NAME)[1]
            self.AtomNum = len(atoms)
            self.Elem, self.ENum = self.count_elements(atoms[:, 1])
            BasisFile = PATH + "B_%s.txt" % NAME
            if os.path.exists(BasisFile):
                self.Basis = self.read_basis_name(BasisFile, self.Elem)
            else:
                self.Basis = '6-31g*'
        return self.str()

    @classmethod
    def get_from_data(cls, PATH, NAME, DATA, BMARK=0):
        '''Get a new mol_slice object from given data'''
        try:
            Elem, ENum = cls.count_elements(DATA[:, 1])
            Stat = cls.chk_files(PATH, NAME)
            if BMARK:
                Basis = basis_compress(BMARK, Elem)
            else:
                BasisFile = PATH + "B_%s.txt" % NAME
                if os.path.exists(BasisFile):
                    Basis = cls.read_basis_name(BasisFile, Elem)
                else:
                    Basis = '6-31g*'
            ms = cls(len(DATA), PATH, NAME, Elem, ENum, Stat, Basis)
            ms.Data = DATA
        except BaseException as e:
            print('Fail to import from data as molslc %s' % NAME)
            raise e
        return ms

    @classmethod
    def get_from_file(cls, FULLNAME, BMARK=0):
        '''Get a new mol_slice object from file'''
        try:
            PATH, FILE = os.path.split(os.path.abspath(FULLNAME))
            PATH = PATH + os.sep
            DATA = read_file(FULLNAME)[1]
            assert FILE[-4:] == '.csv' and FILE.find('atom_list_') == 0
            NAME = FILE[10:-4]
        except BaseException as e:
            print('Fail to import from file %s' % FULLNAME)
            raise e
        return cls.get_from_data(PATH, NAME, DATA, BMARK)

    @classmethod
    def get_from_str(cls, LINE):
        '''Get a new mol_slice object from string'''
        StrList = LINE.split()
        # print(StrList)
        EList = StrList[5:]
        Elem = atom_no(EList[::2])
        ENum = np.array(EList[1::2]).astype(int)
        return cls(int(StrList[0]), StrList[1], StrList[2], Elem, ENum, int(StrList[4], 16), StrList[3])

    def get_coor0center(self, Update=False):
        '''Calculate zero-centerd coordinates'''
        if Update or not len(self.Coor0Center):
            Atoms = self.get_data()
            self.Coor0Center = Atoms[:, 3:6] - np.mean(Atoms[:, 3:6], axis=0)
        return self.Coor0Center

    def get_disinfo(self, Update=False):
        '''Distance information of current slice.

        DisInfo
        --------
            (N,2) Array, N is the atom number
            Column:
                0:      distance to center
                1:      order sorted by distance first, then element
        '''
        if Update or not len(self.DisInfo):
            Atoms = self.get_data()
            Coor = self.get_coor0center()
            Dis = np.sqrt(np.sum(Coor**2, axis=1))
            self.DisInfo = [Dis, np.lexsort([Dis, Atoms[:, 1]])]
        return self.DisInfo

    def get_sortorder(self, Update=False):
        '''Atom order sorted by the distance.

        SortOrder
        --------
            (N,4) Array, N is the atom number
            Column:
                0:      atom number or id
                1:      element number (the position in self.Elem)
                2:      number of atoms in the same section
                3:      section number of distance, smaller means closer distance to the center
        '''
        if Update or not len(self.SortOrder):
            Atoms = self.get_data()
            Coor = Atoms[:, 3:6] - np.mean(Atoms[:, 3:6], axis=0)

            SortOrder = -np.ones((len(Atoms), 4)).astype(int)
            SortOrder[:, 0] = np.arange(len(Atoms))
            Dis = np.sqrt(np.sum(Coor**2, axis=1))
            for i, Ele in enumerate(self.Elem):
                EleIndex = SortOrder[(Atoms[:, 1] == Ele), 0]
                SortOrder[EleIndex, 1] = i
                EleDis = Dis[EleIndex]
                if len(EleDis) < 1:
                    print('0 EleDis for:', self.str())
                Num, Sec = sep_array(EleDis, Tolerance=2 * Coor_Tol)
                for n, s in zip(Num, Sec):
                    pos = np.where((EleDis >= s[0]) & (EleDis <= s[1]))[0]
                    SortOrder[EleIndex[pos], 2] = n
                for n in np.unique(Num):
                    ix_samenum = EleIndex[np.where(SortOrder[EleIndex, 2] == n)[0]]
                    SortOrder[ix_samenum[np.argsort(Dis[ix_samenum])[::-1]], 3] = np.arange(len(ix_samenum)) // n + 1
            self.SortOrder = SortOrder[np.lexsort([SortOrder[:, 1], SortOrder[:, 3], SortOrder[:, 2]]), :]
        return self.SortOrder

    def std_coor(self):
        '''Standard coordinates'''
        Atoms = copy.copy(self.get_data())
        Coor = Atoms[:, 3:6] - np.mean(Atoms[:, 3:6], axis=0)

        SortOrder = self.get_sortorder()
        UniqueOrder = SortOrder[SortOrder[:, 2] == 1, 0]
        pick = [UniqueOrder[0]]
        for ix in UniqueOrder[1:]:
            if len(pick) == 1:
                if not co_line(Coor[pick[0]], Coor[ix]):
                    pick.append(ix)
            elif len(pick) == 2:
                if not co_plane(Coor[[pick[0], pick[1], ix]]):
                    pick.append(ix)
            else:
                break
        if len(pick) < 3:
            print('ERROR: Fail to find 3 unique atoms!\nStandard form not found.\n')
            return
        # print(Coor[pick[:2]])
        ex = Coor[pick[0]] / LA.norm(Coor[pick[0]])
        ey = Coor[pick[1]] - Coor[pick[1]].dot(ex) * ex
        ey = ey / LA.norm(ey)
        ez = np.cross(ex, ey)
        RotMtx = np.array([ex, ey, ez])  # Rotating matrix from coorinate in xyz to new (ex,ey,ez)
        Atoms[:, 3:6] = Coor.dot(RotMtx.T)
        return Atoms

    @staticmethod
    def test_mtx(PickTarget, PickSource, AtomTarget, AtomSource):
        """
        Test whether AtomSource is same with AtomTarget.

        This function first calculate the trasforming matrix from PickSource to PickTarget, then use this matrix to reorder AtomSource,
        and finally calculate the trasforming matrix from AtomSource to AtomTarget.

        Parameters
        ----------
            PickTarget: (n,3) ndarray
            PickSource: (n,3) ndarray, PickTarget and PickSource should be one-to-one correspondence.
            AtomTarget: (N,6) ndarray, standard atom format required
            AtomSource: (N,6) ndarray, standard atom format required

        Standard Atom Format
        --------
        Atoms is (m,n) ndarray, and n >= 6.

        Column:
            0:      atom number or id, not always useful
            1:      element type (1~118)
            2:      atom id or distance
            3~5:    coordinate
            6:      weight (0~1)

        Returns
        -------
            issame:     bool
            rotatemtx:  when issame is True: (3,3) ndarray, the rotating matrix; else: 0
        """
        try:
            # RotateMtx=LA.lstsq(PickSource,PickTarget,rcond=None)[0].T
            RotateMtx = LA.lstsq(PickSource, PickTarget)[0].T
        except Exception:  # LinAlgError:
            return False, 0
        RotateMtx[:3, :3] = form_rot_mtx(RotateMtx, True)
        # print('Rot_Mtx1:',RotateMtx)
        # print('C_Pick:',PickTarget)
        # print('C_Test:',PickSource)
        # print('C_Testr:',np.dot(PickSource,RotateMtx.T))
        # print('Err:',np.dot(PickSource,RotateMtx.T)-PickTarget)

        CoorSource = AtomSource[:, 3:6]
        CoorTarget = AtomTarget[:, 3:6]
        # NewAtom=np.dot(np.hstack([CoorSource,np.ones((CoorSource.shape[0],1))]),RotateMtx.T)
        NewAtom = np.dot(CoorSource, RotateMtx.T)

        TestOrder = np.array([np.argmin(np.sum((NewAtom - test)**2, axis=1)) for test in CoorTarget]).astype(int)
        if not (AtomTarget[:, 1] == AtomSource[TestOrder, 1]).all():
            print('Fail at mtx:', RotateMtx, flush=True)
            return False, 0

        # RotOld=RotateMtx.copy()
        # TestCoor=np.hstack([CoorSource[TestOrder,:],np.ones((CoorSource.shape[0],1))])
        TestCoor = CoorSource[TestOrder, :]
        try:
            RotateMtx = LA.lstsq(TestCoor, CoorTarget)[0].T
        except Exception:  # LinAlgError:
            return False, 0
        # print('Rot_Mtx3:',RotateMtx)
        RotateMtx[:3, :3] = form_rot_mtx(RotateMtx)
        # print('Rot_Mtx4:',RotateMtx)
        # print('Err_Old:',CoorTarget-np.dot(TestCoor,RotOld.T))
        # print('Err:',CoorTarget-np.dot(TestCoor,RotateMtx.T))
        if cmp_atoms(CoorTarget, np.dot(TestCoor, RotateMtx.T)):
            # print('Good')
            return True, RotateMtx[:3, :3]
        else:
            # print('Fail')
            return False, 0

    # @classmethod
    def same(self, other, CheckBasis=True, Tolerance=Coor_Tol, PoolSize=Job.CPUAvail, Debug=False):
        '''Whether two mol_slice are the same. Also return the rotating matrix if True'''
        assert isinstance(other, mol_slice), "ERROR: Data type not match!\n"
        if (self.AtomNum == other.AtomNum) and np.all(self.Elem == other.Elem) and np.all(self.ENum == other.ENum):
            if CheckBasis and (self.Basis != other.Basis):
                return False, 0
            A0 = self.get_data()
            A1 = other.get_data()
            Atoms0 = copy.copy(A0)
            Atoms1 = copy.copy(A1)
            # Coor0 = Atoms0[:, 3:6] = A0[:, 3:6] - np.mean(A0[:, 3:6], axis=0)
            # Coor1 = Atoms1[:, 3:6] = A1[:, 3:6] - np.mean(A1[:, 3:6], axis=0)
            Coor0 = Atoms0[:, 3:6] = self.get_coor0center()
            Coor1 = Atoms1[:, 3:6] = other.get_coor0center()
            if np.all(Atoms0[:, 1] == Atoms1[:, 1]):
                if np.allclose(Coor0, Coor1, atol=Tolerance):
                    return True, np.diag([1, 1, 1])
                try:
                    # RotateMtx=LA.lstsq(Atoms1[:,3:6],Atoms0[:,3:6],rcond=None)[0].T
                    RotateMtx = LA.lstsq(Coor1, Coor0)[0].T
                    RotateMtx[:3, :3] = form_rot_mtx(RotateMtx)
                    # print('Rot_Mtx00:',RotateMtx)
                    if cmp_atoms(Coor0, np.dot(Coor1, RotateMtx.T)):
                        return True, RotateMtx[:3, :3]
                except Exception:  # LinAlgError:
                    pass
            Dis0, DisIdx0 = self.get_disinfo()  # np.sqrt(np.sum(Coor0**2, axis=1))
            Dis1, DisIdx1 = other.get_disinfo()  # np.sqrt(np.sum(Coor1**2, axis=1))
            if Debug:
                # print('Dis0, Dis1:')
                # for i in range(min(30, len(Dis0))):
                #     print(i, ',', Dis0[i], ',', Dis1[i])
                print('set distance done.')
            if not np.allclose(Dis0[DisIdx0], Dis1[DisIdx1], atol=Tolerance):
                return False, 0
            if Debug:
                print('distance pass.')

            order0 = self.get_sortorder()
            NoIdx2DisIdx = np.argsort(DisIdx0)
            order1 = np.hstack([DisIdx1[NoIdx2DisIdx[order0[:, 0]]].reshape((-1, 1)), order0[:, 1:]])
            # DisOrder = []
            # for i in order0[:, 0]:
            #     BestSeq = np.argsort(abs(Dis1 - Dis0[i]) + 10 * abs(Atoms1[:, 1] - Atoms0[i, 1]))
            #     for b in BestSeq:
            #         if b not in DisOrder:
            #             DisOrder.append(b)
            #             break
            # DisOrder = np.array([np.argmin(abs(Dis1 - Dis0[_]) + 10*abs(Atoms1[:, 1] - Atoms0[_, 1])) for _ in order0[:, 0]]).astype(int)
            # order1 = np.hstack([np.array(DisOrder).reshape((-1, 1)), order0[:, 1:]])
            if Debug:
                # print('order0, order1:')
                # for i in range(min(20, len(order0))):
                #     print(order0[i], ',', order1[i])
                print('orderdis0, orderdis1:')
                for i in range(len(order0)):
                    err = abs(Dis0[order0[i, 0]] - Dis1[order1[i, 0]])
                    if err > 0.8 * Tolerance:
                        shownum = order0[i, 2] - 1
                        for j in range(max(0, i - shownum), min(i + shownum + 1, len(order0))):
                            print(j, ',', order0[j], ',', order1[j])
                            print(Dis0[order0[j, 0]], ',', Dis1[order1[j, 0]], ',', abs(Dis0[order0[j, 0]] - Dis1[order1[j, 0]]))
                print('set disorder done.')
            if not np.allclose(Dis0[order0[:, 0]], Dis1[order1[:, 0]], atol=1.5 * Tolerance):
                return False, 0
            if Debug:
                print('disorder pass.')

            CoPln, Pairs = get_sets_from_order(order0, order1, Coor0, Coor1)  #, Debug=Debug)
            if len(Pairs) == 1:
                return False, 0
            if Debug:
                print('set pairs done.\nPairs:', len(Pairs))
                if len(Pairs) < 4:
                    print(Pairs)
                else:
                    for i in range(min(20, len(Pairs))):
                        print(Pairs[i])
            # exit()

            PickCoor = Coor0[Pairs[0], :]
            if CoPln:
                AddNo = not_co_line_all(PickCoor)
                if AddNo:
                    AddCoor = PickCoor[AddNo[0], :] + np.cross(PickCoor[AddNo[1], :] - PickCoor[AddNo[0], :], PickCoor[AddNo[2], :] - PickCoor[AddNo[0], :])
                    PickCoor = np.vstack([PickCoor, AddCoor.reshape([1, -1])])
                else:
                    print('WARNING: Test atoms are in a line! Result may be wrong!')
                    print('My ', self.Key.Str)
                    print('At ', other.Key.Str)
                    print('AddNo: ', AddNo)
                    print('Pick:\n', PickCoor, flush=True)
            if Debug:
                print('set pickcoor done.')
            if len(Pairs) == 2:
                TestCoor = Coor1[Pairs[1], :]
                return self.test_mtx(PickCoor, TestCoor, Atoms0, Atoms1)
            else:
                if PoolSize > 1:
                    pool = mp.Pool(PoolSize)
                    res = []
                    for p in Pairs[1:]:
                        # TestCoor = EleCoor1[p, :]
                        TestCoor = Coor1[p, :]
                        # TestCoor=EleC1[p,:]
                        if CoPln and AddNo:
                            AddCoor = TestCoor[AddNo[0], :] + np.cross(TestCoor[AddNo[1], :] - TestCoor[AddNo[0], :], TestCoor[AddNo[2], :] - TestCoor[AddNo[0], :])
                            TestCoor = np.vstack([TestCoor, AddCoor.reshape([1, -1])])
                        # TestCoor=np.hstack([TestCoor,np.ones((TestCoor.shape[0],1))])
                        res.append(pool.apply_async(self.test_mtx, (PickCoor, TestCoor, Atoms0, Atoms1)))
                        # res.append(pool.apply_async(self.test_mtx,(PickCoor,TestCoor,A0,A1,)))
                    pool.close()
                    pool.join()
                    chk = [x.get() for x in res]
                else:
                    chk = []
                    for p in Pairs[1:]:
                        TestCoor = Coor1[p, :]
                        if CoPln and AddNo:
                            AddCoor = TestCoor[AddNo[0], :] + np.cross(TestCoor[AddNo[1], :] - TestCoor[AddNo[0], :], TestCoor[AddNo[2], :] - TestCoor[AddNo[0], :])
                            TestCoor = np.vstack([TestCoor, AddCoor.reshape([1, -1])])
                        chk.append(self.test_mtx(PickCoor, TestCoor, Atoms0, Atoms1))
                if Debug:
                    print('testmtx done.')
                s = []
                for i, cc in enumerate(chk):
                    if cc[0]:
                        s.append(cc)
                        print('No:', i, '\nmtx:', cc[1], flush=True)
                if (len(s)):
                    print('match %d in %d' % (len(s), len(chk)), flush=True)
                    return s[0]
                return False, 0
        else:
            return False, 0
