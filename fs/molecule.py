#!/usr/bin/env python
'''
Author: Ifsoul
Date: 2020-08-29 15:29:05
LastEditTime: 2021-12-07 17:41:33
LastEditors: Ifsoul
Description: Functions for DNA and proteins
'''
import numpy as np
import math
import os
import _pickle as pk
from scipy import linalg as LA
from copy import deepcopy

from .data import deg2rad, Elements, Elem_Radius, AA_List, NA_List, AA2single, NA2single, Base_List, Base2num, RNABase2num, RNABase_List, Base64, Base64_Dict, NA_base, DNA_pair, RNA_pair, DNA_base2name, RNA_base2name, helix_dict
from .core import sep_num_abc, atom_no, atom_type, find_all, get_ext, bond_chk
from .fileIO import IO_File, read_file, read_txt_lines, load_fsz_thread, load_xyz, save_xyz, info_file


class atom(object):
    '''atom'''

    def __init__(self, ID=-1, NAME='X', ELEM=0, COOR=np.zeros((3,)), NO=-1, RESID=-1, SEQNO=0, OCCU=1.0, BFACT=0):
        self.Id = ID
        self.No = NO
        self.Virt = False
        self.Name = NAME
        self.Elem = ELEM
        self.Coor = COOR
        self.Bonds = []
        self.HBonds = []
        self.Near = []
        self.ResId = RESID
        self.SeqNo = SEQNO
        self.Occu = OCCU
        self.BFact = BFACT
        self.Dis = -1

    def set_dis(self, Source, Gather, Dis=0, BOX=()):
        """set distance of each atom"""
        assert len(Gather) >= len(Source), "ERROR: Length of Gather too small!\n"
        if Gather[self.Id] == -1 or Gather[self.Id] > Dis:
            Gather[self.Id] = Dis
            for ids in self.Bonds:
                if BOX and (ids not in BOX):
                    continue
                Source[ids].set_dis(Source, Gather, Dis + 1, BOX)

    def show(self):
        """print atom information"""
        print('Atom: %d\tName: %s\tElem: %s\nResId: %d\tSeqNo: %d\tNo: %d\tDis: %g' % (
            self.Id,
            self.Name,
            atom_type(self.Elem),
            self.ResId,
            self.SeqNo,
            self.No,
            self.Dis,
        ))
        print('Coor: %s' % ('\t'.join(['%g' % x for x in self.Coor])))
        print('Bonds: %s' % ('\t'.join(['%d' % x for x in self.Bonds])))
        print('HBonds: %s' % ('\t'.join(['%d' % x for x in self.HBonds])))

    def bond_chain(self, Source, level=1):
        """bonds (and neighbor bonds) of atom"""
        if level <= 1:
            return self.Bonds
        else:
            chain = []
            for ids in self.Bonds:
                chain.extend(Source[ids].bond_chain(Source, level - 1))
            return chain

    @classmethod
    def copy(cls, src_atm):
        """Get a new atom by copy"""
        new_atom = cls()
        new_atom.__dict__.update(deepcopy(src_atm.__dict__))
        return new_atom


class residue(object):
    '''residue'''

    def __init__(self, ID=-1, NAME='X', TYPE='NULL', NO=-1, CHAINID=-1, SEQNO=0):
        self.Id = ID
        self.No = NO
        self.Name = NAME
        self.Type = TYPE
        self.AtomList = []
        self.Links = []
        self.HLinks = []
        self.Center = []
        self.Near = []
        self.ChainId = CHAINID
        self.SeqNo = SEQNO
        self.Dis = -1

    def set_dis(self, Source, dis=0, DoubleStrand=True):
        """set distance of each residue"""
        if self.Dis == -1 or self.Dis > dis:
            self.Dis = dis
            for ids in self.Links:
                Source[ids].set_dis(Source, dis + 1, DoubleStrand)
            if DoubleStrand:
                for ids in self.HLinks:
                    if ids not in self.Links:
                        Source[ids].set_dis(Source, dis, DoubleStrand)

    def set_dis_fast(self, Source, dis=0, DoubleStrand=True, TailIds=[]):
        """fast set distance of each residue"""
        CurrentLevel = []
        NextLevel = [self.Id]
        TempList = []
        self.Dis = dis
        while True:
            if DoubleStrand:
                while True:
                    CurrentLevel.extend(NextLevel)
                    for rid_set in NextLevel:
                        res = Source[rid_set]
                        for rid in res.HLinks:
                            if rid not in res.Links and (Source[rid].Dis == -1 or Source[rid].Dis > dis):
                                Source[rid].Dis = dis
                                TempList.append(rid)
                    NextLevel, TempList = TempList, []
                    if not NextLevel:
                        break
            if TailIds and dis == 0:
                for rid_set in CurrentLevel:
                    for rid in Source[rid_set].Links:
                        if rid in TailIds:
                            continue
                        if (Source[rid].Dis == -1 or Source[rid].Dis > dis):
                            Source[rid].Dis = dis + 1
                            NextLevel.append(rid)
            else:
                for rid_set in CurrentLevel:
                    for rid in Source[rid_set].Links:
                        if (Source[rid].Dis == -1 or Source[rid].Dis > dis):
                            Source[rid].Dis = dis + 1
                            NextLevel.append(rid)
            dis += 1
            CurrentLevel, NextLevel = NextLevel, []
            if not CurrentLevel:
                break

    @classmethod
    def copy(cls, src_rsd):
        """Get a new residue by copy"""
        new_residue = cls()
        new_residue.__dict__.update(deepcopy(src_rsd.__dict__))
        return new_residue


class chain(object):
    '''chain'''

    def __init__(self, ID=-1, NAME='X', CHAINNO=0):
        self.Id = ID
        self.Name = NAME
        self.ResList = []
        self.ChainNo = CHAINNO

    @classmethod
    def copy(cls, src_chain):
        """Get a new chain by copy"""
        new_chain = cls()
        new_chain.__dict__.update(deepcopy(src_chain.__dict__))
        return new_chain


class molecule(object):
    '''molecule'''

    def __init__(self, NAME='X', TYPE='NULL'):
        self.Name = NAME
        self.Type = TYPE
        self.Atoms = []
        self.Residues = []
        self.Chains = []

    def show(self):
        """print molecule information"""
        print('Mol Name: %s\tType: %s' % (
            self.Name,
            self.Type,
        ))
        print('%d Atoms' % (len(self.Atoms)))
        print('%d Residues' % (len(self.Residues)))
        print('%d Chains' % (len(self.Chains)))

    def set_bonds(self, ProteinVirtH=False):
        """set bonds of atoms in molecule"""
        t = self.Type.upper()
        DS_DNA = t.startswith('DSDNA') or t.startswith('DSRNA')
        SS_DNA = t.startswith('SSDNA') or t.startswith('SSRNA')
        CoorTot = np.array([a.Coor for a in self.Atoms])
        EleTot = np.array([a.Elem for a in self.Atoms])
        MaxRad = max([Elem_Radius[e - 1] for e in np.unique(EleTot)])
        VirtList = []
        Ntot = len(self.Atoms)
        for i, res in enumerate(self.Residues):
            CoorCenter = np.zeros((3,))
            for Aid in res.AtomList:
                CoorCenter += self.Atoms[Aid].Coor
            self.Residues[i].Center = CoorCenter / len(res.AtomList)
        CenterTot = np.array([r.Center for r in self.Residues])
        MAX_RES_DIS_SQ = 256
        for r in self.Residues:
            dis = np.sum((CenterTot[r.Id + 1:, :] - r.Center)**2, axis=1)
            select = np.where(dis < MAX_RES_DIS_SQ)[0]
            for j in select:
                s = self.Residues[j + r.Id + 1]
                if r.Id not in s.Near:
                    r.Near.append(s.Id)
                if s.Id not in r.Near:
                    s.Near.append(r.Id)
        for a in self.Atoms:
            ResA = self.Residues[a.ResId]
            NearList = ResA.AtomList.copy()
            for rid in ResA.Near:
                NearList.extend(self.Residues[rid].AtomList)
            NearList = [_ for _ in set(NearList) if _ > a.Id]
            dis2 = np.sum((CoorTot[NearList, :] - a.Coor)**2, axis=1)
            MaxDis2 = 4.85 * (Elem_Radius[a.Elem - 1] + MaxRad)**2
            select = np.where(dis2 < MaxDis2)[0]
            for j, mydis in zip(select, np.sqrt(dis2[select])):
                b = self.Atoms[NearList[j]]
                # if a.Id ==27 or b.Id ==27:
                #     print(f'a:{a.Id} {a.Elem} b:{b.Id} {b.Elem} dis:{mydis}')
                if bond_chk(a.Elem, b.Elem, mydis):
                    # if a.Id ==27 or b.Id ==27:
                    #     print('pass')
                    if b.Id not in a.Bonds:
                        a.Bonds.append(b.Id)
                    if a.Id not in b.Bonds:
                        b.Bonds.append(a.Id)
                    if a.ResId != b.ResId:
                        ResA = self.Residues[a.ResId]
                        ResB = self.Residues[b.ResId]
                        NOT_LINKED = False
                        if b.ResId not in ResA.Links:
                            ResA.Links.append(b.ResId)
                            NOT_LINKED = True
                        if a.ResId not in ResB.Links:
                            ResB.Links.append(a.ResId)
                            NOT_LINKED = True
                        if ProteinVirtH and NOT_LINKED and not (DS_DNA or SS_DNA):
                            Vab = (a.Coor - b.Coor)
                            Vab /= LA.norm(Vab)
                            CoorA = a.Coor - Vab * (Elem_Radius[0] + Elem_Radius[a.Elem - 1])
                            CoorB = b.Coor + Vab * (Elem_Radius[0] + Elem_Radius[b.Elem - 1])
                            Atmp = atom(len(VirtList) + Ntot, 'HVIR', 1, CoorA, -1, a.ResId, len(ResA.AtomList))
                            Atmp.Virt = True
                            Atmp.Bonds.append(a.Id)
                            Atmp.HBonds.append(b.Id)
                            a.Bonds.append(Atmp.Id)
                            ResA.AtomList.append(Atmp.Id)
                            Btmp = atom(len(VirtList) + Ntot + 1, 'HVIR', 1, CoorB, -1, b.ResId, len(ResB.AtomList))
                            Btmp.Virt = True
                            Btmp.Bonds.append(b.Id)
                            Btmp.HBonds.append(a.Id)
                            b.Bonds.append(Btmp.Id)
                            ResB.AtomList.append(Btmp.Id)
                            VirtList.extend([Atmp, Btmp])
                else:
                    if b.Id not in a.Near:
                        a.Near.append(b.Id)
                    if a.Id not in b.Near:
                        b.Near.append(a.Id)
        self.Atoms.extend(VirtList)

    def set_hbonds(self):
        """set H-bonds of atoms in molecule"""
        for a in self.Atoms:
            if a.Elem != 1 or a.Virt:
                continue
            MyNBond = len(a.Bonds)
            if MyNBond != 1:
                a.show()
                raise ValueError("ERROR: Zero or multiple bonds of H atom %d!\n" % a.Id)
            b = self.Atoms[a.Bonds[0]]
            if b.Elem not in [6, 7, 8, 9, 16, 17]:
                continue
            for j in a.Near:
                c = self.Atoms[j]
                if c.Elem not in [7, 8, 9, 16, 17]:
                    continue
                r_ac = a.Coor - c.Coor
                r_ab = a.Coor - b.Coor
                r_cb = c.Coor - b.Coor
                # if LA.norm(r_ac)>2.5*(Elem_Radius[0]+Elem_Radius[c.Elem-1]): continue
                if LA.norm(r_ac) > 2.2 * (Elem_Radius[0] + Elem_Radius[c.Elem - 1]):
                    continue
                if r_cb.dot(r_ab) < 0.866025403 * (LA.norm(r_cb) * LA.norm(r_ab)):
                    continue
                # if r_cb.dot(r_ab)<0.93969262*(LA.norm(r_cb)*LA.norm(r_ab)): continue
                a.HBonds.append(c.Id)
                c.HBonds.append(a.Id)
                if a.ResId != c.ResId:
                    if c.ResId not in self.Residues[a.ResId].HLinks:
                        self.Residues[a.ResId].HLinks.append(c.ResId)
                    if a.ResId not in self.Residues[c.ResId].HLinks:
                        self.Residues[c.ResId].HLinks.append(a.ResId)

    def set_distance(self):
        """not used"""
        pass

    def gmx2amber(self):
        """transform gromacs DNA into amber DNA"""
        assert self.Type.upper() in ['DSDNA', 'SSDNA']
        for res in self.Residues:
            assert res.Type in NA_List
            ix = NA_List.index(res.Type)
            newname = NA_List[5 + ix % 5].upper()
            names = [self.Atoms[a].Name for a in res.AtomList]
            if 'H3T' in names and 'H5T' in names:
                newname = newname + 'N'
            elif 'H3T' in names:
                newname = newname + '3'
            elif 'H5T' in names:
                newname = newname + '5'
            res.Name = newname
            if res.Type.upper() in ['THY', 'DT']:
                for a in res.AtomList:
                    atm = self.Atoms[a]
                    if atm.Name == 'C5M':
                        atm.Name = 'C7'
                    if atm.Name == 'H51':
                        atm.Name = 'H71'
                    if atm.Name == 'H52':
                        atm.Name = 'H72'
                    if atm.Name == 'H53':
                        atm.Name = 'H73'

    def my_name(self):
        """default name"""
        t = self.Type.upper()
        Head_Dict = {'DSDNA': 'DD', 'SSDNA': 'SD', 'DSRNA': 'DR', 'SSRNA': 'SR', 'PROTEIN': 'PRO'}
        assert t in Head_Dict, 'ERROR: Unkown type of molecule!'
        Head = Head_Dict[t]
        single_dict = AA2single if Head == 'PRO' else NA2single
        seq = []
        for rno in self.Chains[0].ResList:
            r = self.Residues[rno]
            seq.append(single_dict[r.Type.capitalize()])
        return Head + '-' + ''.join(seq[:8])

    def save_as_file(self, FileName, SaveXYZ=True):
        """Save molecule as .pdb or .fsmol file"""
        # Name, Ext = get_ext(FileName, ReturnName=True)
        Ext = get_ext(FileName)
        if Ext not in ('pdb', 'pqr', 'fsmol'):
            print('WARNING: Unknown file format (%s)!' % Ext)
            # Name, Ext = FileName, 'fsmol'
            Ext = 'fsmol'
            FileName = FileName + '.fsmol'
            print('Save file as: %s' % FileName)
        if Ext in ('pdb', 'pqr'):
            pdb_file(FileName).save(Mol=self, BOND=False)
        else:
            with open(FileName, 'wb') as f:
                pk.dump(self, f)
        if SaveXYZ:
            XyzData = []
            for atm in self.Atoms:
                XyzData.append([atm.Id, atm.Elem, 0, atm.Coor[0], atm.Coor[1], atm.Coor[2]])
            save_xyz(FileName + '.xyz', np.array(XyzData))

    @classmethod
    def load_from_file(cls, FileName,verbose=False):
        """Get a new molecule from .pdb or .fsmol file"""
        # Name, Ext = get_ext(FileName, ReturnName=True)
        Ext = get_ext(FileName)
        if Ext in ('pdb', 'pqr'):
            mol = pdb_file(FileName).load()
            XyzFile = FileName + '.xyz'
            if os.path.isfile(XyzFile):
                if verbose:
                    print('Updating coordinates based on file %s.' % XyzFile)
                XyzData = load_xyz(XyzFile)
                for i in range(len(mol.Atoms)):
                    assert mol.Atoms[i].Elem == int(XyzData[i, 1])
                    mol.Atoms[i].Coor = XyzData[i, 3:6]
            return mol
        elif Ext == 'fsmol':
            with open(FileName, 'rb') as f:
                return pk.load(f)
        elif Ext in ('csv', 'xyz'):
            coor_data = read_file(FileName)[1]
            return cls.load_by_coor_data(coor_data)
        else:
            raise ValueError('Unknown file format (%s)!' % Ext)

    @classmethod
    def load_by_coor_data(cls, CoorData):
        """Get a new molecule from Data"""
        NewMol = cls()
        Ctmp = chain(len(NewMol.Chains))
        NewMol.Chains.append(Ctmp)
        Rtmp = residue(len(NewMol.Residues), NO=0)
        Rtmp.ChainId = Ctmp.Id
        Rtmp.SeqNo = len(Ctmp.ResList)
        Ctmp.ResList.append(Rtmp.Id)
        NewMol.Residues.append(Rtmp)
        for atm_data in CoorData:
            eleno = int(atm_data[1])
            Atmp = atom(len(NewMol.Atoms), atom_type(eleno), eleno, atm_data[3:6], int(atm_data[0]))
            Atmp.ResId = Rtmp.Id
            Atmp.SeqNo = len(Rtmp.AtomList)
            Rtmp.AtomList.append(Atmp.Id)
            NewMol.Atoms.append(Atmp)
        return NewMol

    @classmethod
    def copy(cls, src_mol):
        """Get a new molecule by copy"""
        new_mol = cls(src_mol.Name, src_mol.Type)
        for src_atm in src_mol.Atoms:
            new_mol.Atoms.append(atom.copy(src_atm))
        for src_rsd in src_mol.Residues:
            new_mol.Residues.append(residue.copy(src_rsd))
        for src_chn in src_mol.Chains:
            new_mol.Chains.append(chain.copy(src_chn))
        return new_mol

    def add_new_residues(self, src_mol, src_rsd_list, tag_chn_idx=0):
        """Add new residues from another molecule into target chain"""
        if tag_chn_idx >= len(self.Chains):
            for i in range(len(self.Chains), tag_chn_idx + 1):
                Ctmp = chain(len(self.Chains), f'{i}')
                Ctmp.No = Ctmp.Id + 1
                self.Chains.append(Ctmp)
        Ctmp = self.Chains[tag_chn_idx]
        for src_rsd_idx in src_rsd_list:
            src_rsd = src_mol.Residues[src_rsd_idx]
            Rtmp = residue(len(self.Residues), src_rsd.Name, NO=src_rsd.No)
            Rtmp.Type = src_rsd.Type
            Rtmp.ChainId = Ctmp.Id
            Rtmp.SeqNo = len(Ctmp.ResList)
            Ctmp.ResList.append(Rtmp.Id)
            self.Residues.append(Rtmp)
            for atm_no in src_rsd.AtomList:
                src_atm = src_mol.Atoms[atm_no]
                Atmp = atom(
                    len(self.Atoms),
                    src_atm.Name,
                    src_atm.Elem,
                    src_atm.Coor,
                    src_atm.No,
                    OCCU=src_atm.Occu,
                    BFACT=src_atm.BFact,
                )
                Atmp.ResId = Rtmp.Id
                Atmp.SeqNo = len(Rtmp.AtomList)
                Rtmp.AtomList.append(Atmp.Id)
                self.Atoms.append(Atmp)

    @classmethod
    def new_by_copy_redidues(cls, src_mol, src_rsd_list, NewName=None):
        """Get new molecule by copy residues from another molecule"""
        if NewName is None:
            NewName = src_mol.Name + ' rsd ' + ' '.join(['%d' % _ for _ in src_rsd_list])
        new_mol = cls(NewName, src_mol.Type)
        Ctmp = chain(len(new_mol.Chains), 'A')
        Ctmp.No = Ctmp.Id + 1
        new_mol.Chains.append(Ctmp)
        new_mol.add_new_residues(src_mol, src_rsd_list)
        return new_mol

    def get_full_coor(self):
        """Get coordinates of all atoms in current molecule"""
        Coors = []
        for i, atm in enumerate(self.Atoms):
            Coors.append([i, atm.Elem, atm.Id, *atm.Coor.tolist()])
        return np.array(Coors)

    def update_coor(self, new_coors, idx_col=0, elem_col=1, coor_col=3):
        """Get coordinates of all atoms in current molecule"""
        for coor in new_coors:
            my_idx = int(coor[idx_col])
            my_elem = int(coor[elem_col])
            atm = self.Atoms[my_idx]
            if my_elem != atm.Elem:
                raise RuntimeError(f'ERROR: Mismatch of element type for atom {atm.Id} (expect {atm.Elem} but given {my_elem})!')
            atm.Coor = np.array(coor[coor_col:coor_col + 3])
        return self

    def merge_all_residues(self):
        """Merge all atoms into one residue"""
        if len(self.Residues) > 1:
            rsd0 = self.Residues[0]
            for rsd in self.Residues[1:]:
                for atm_id in rsd.AtomList:
                    atm = self.Atoms[atm_id]
                    atm.ResId = 0
                    atm.SeqNo = len(rsd0.AtomList)
                    rsd0.AtomList.append(atm_id)
            self.Residues = [rsd0]
        if len(self.Chains) > 1:
            self.Chains[0].ResList = [0]
            self.Chains = [self.Chains[0]]


class ring(residue):
    '''ring(residue)'''

    def __init__(self, ID=-1, NAME='X', TYPE='NULL', NO=-1, CHAINID=-1, SEQNO=0):
        super(ring, self).__init__(ID, NAME, TYPE, NO, CHAINID, SEQNO)


class hex_mol(molecule):
    '''hex_mol(molecule)'''

    def __init__(self, NAME='X', TYPE='NULL'):
        super(hex_mol, self).__init__(NAME, TYPE)
        self.Rings = []


class pdb_file(IO_File):
    '''pdb file'''

    def __init__(self, FILENAME):
        super().__init__(FILENAME)
        self.Mol = molecule(self.Proj)

    def load(self):
        '''load molecule from file'''
        assert os.path.isfile(self.Name), "ERROR: Cannot find file %s!\n" % self.Name
        assert self.Ext.lower() in ('pdb', 'pqr'), "ERROR: Wrong file format %s!\n" % self.Ext
        self.Lines = [x.strip() for x in open(self.Name, 'rt').readlines()]
        NewChain = True
        NewRes = True
        for line in self.Lines:
            Key = line[:6].strip()
            if Key == 'ATOM' or Key == 'HETATM':
                name = line[12:16].strip()
                elename = line[12:14].strip()
                ele = elename[0].upper() + elename[1].lower() if len(elename) > 1 else elename[0].upper()
                if ele[0] == 'H' and len(ele) == 2 and line[14:16].isdigit():
                    ele = 'H'
                if ele not in Elements[:54] and ele[0] in Elements:
                    ele = ele[0]
                try:
                    AtomNo = int(line[6:12])
                    ResNo = int(line[22:27]) if line[26].isdigit() else int(line[22:26])
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        occu = float(line[54:60])
                        bfact = float(line[60:66])
                    except Exception:
                        x, y, z, occu, bfact = (float(_) for _ in line[30:].split()[:5])
                except Exception as e:
                    print('Exception %s: Fail to read line:\n\t%s' % (e, line))
                Atmp = atom(len(self.Mol.Atoms), name, atom_no(ele), np.array([x, y, z]), AtomNo, OCCU=occu, BFACT=bfact)
                res = line[17:20].strip()
                cname = line[21]
                if not NewChain and (ResNo < self.Mol.Residues[-1].No or cname != self.Mol.Chains[-1].Name):
                    NewChain = True
                if not NewRes and (NewChain or ResNo != self.Mol.Residues[-1].No or res != self.Mol.Residues[-1].Name):
                    NewRes = True
                if NewRes:
                    Rtmp = residue(len(self.Mol.Residues), res, NO=ResNo)
                    res = res[0] + res[1:3].lower()
                    if res in AA_List or res in NA_List or res[:2] in NA_List:
                        Rtmp.Type = res
                    if NewChain:
                        Ctmp = chain(len(self.Mol.Chains), cname)
                        Ctmp.No = Ctmp.Id + 1
                        self.Mol.Chains.append(Ctmp)
                        NewChain = False
                    else:
                        Ctmp = self.Mol.Chains[-1]
                    Rtmp.ChainId = Ctmp.Id
                    Rtmp.SeqNo = len(Ctmp.ResList)
                    Ctmp.ResList.append(Rtmp.Id)
                    self.Mol.Residues.append(Rtmp)
                    NewRes = False
                else:
                    Rtmp = self.Mol.Residues[-1]
                Atmp.ResId = Rtmp.Id
                Atmp.SeqNo = len(Rtmp.AtomList)
                Rtmp.AtomList.append(Atmp.Id)
                self.Mol.Atoms.append(Atmp)
            elif Key == 'CONECT':
                NoList = []
                for k in range(7, 28, 5):
                    stmp = line[k:k + 5]
                    if stmp:
                        NoList.append(int(stmp))
                NoList = list(set(NoList))
                IdList = -1 * np.ones_like(NoList)
                for atm in self.Mol.Atoms:
                    if atm.No in NoList:
                        IdList[NoList.index(atm.No)] = atm.Id
                assert (IdList != -1).all(), "ERROR: Fail to find atom at line:\n\t%s" % line
                atm = self.Mol.Atoms[IdList[0]]
                atm.Bonds.extend([i for i in IdList[1:] if i not in atm.Bonds])
                for i in IdList[1:]:
                    atm = self.Mol.Atoms[i]
                    if IdList[0] not in atm.Bonds:
                        atm.Bonds.append(IdList[0])
            elif Key == 'TER':
                NewChain = True
                NewRes = True
            elif Key == 'END' or Key == 'ENDMDL':
                break
        return self.Mol

    def save(self, Output=None, Mol=None, BOND=True, WARNING=True):
        '''save molecule into file'''
        if not Output:
            Output = self.Name
        ext = get_ext(Output)
        assert ext.lower() in ('pdb', 'pqr'), "ERROR: Wrong file format %s!\n" % ext
        if not Mol:
            Mol = self.Mol
        if WARNING and os.path.isfile(Output):
            print('WARNING: Overwriting existing file %s' % Output)
        with open(Output, 'wt') as f:
            for c in Mol.Chains:
                c_name = c.Name if len(c.Name) == 1 else '%d' % c.Id
                for j in c.ResList:
                    res = Mol.Residues[j]
                    for k in res.AtomList:
                        atm = Mol.Atoms[k]
                        atm_name = atm.Name.upper()
                        if len(atm_name) < 4 and len(atom_type(atm.Elem)) == 1:
                            atm_name = ' ' + atm_name
                        s_atmid = '%5d '%atm.Id if atm.Id<100000 else '%6d'%atm.Id
                        s_resid = '%4d '%res.Id if res.Id<10000 else '%5d'%res.Id
                        s_coorx = '%8.3f'%atm.Coor[0] if -1000<atm.Coor[0]<10000 else '%8.2f'%atm.Coor[0]
                        s_coory = '%8.3f'%atm.Coor[1] if -1000<atm.Coor[1]<10000 else '%8.2f'%atm.Coor[1]
                        s_coorz = '%8.3f'%atm.Coor[2] if -1000<atm.Coor[2]<10000 else '%8.2f'%atm.Coor[2]
                        f.write('ATOM  %6s%-4s%1s%3s %1s%5s   %8s%8s%8s%6.2f%6.2f      %4s%2s%2s\n' % (
                            s_atmid,
                            atm_name,
                            ' ',
                            res.Name.upper(),
                            c_name,
                            s_resid,
                            s_coorx,
                            s_coory,
                            s_coorz,
                            atm.Occu,
                            atm.BFact,
                            ' ',
                            ' ',
                            ' ',
                        ))
                f.write('TER   \n')
            if BOND:
                for a in Mol.Atoms:
                    BList = [x for x in a.Bonds if x > a.Id]
                    if not BList:
                        continue
                    for i in range(0, len(BList), 4):
                        atmlist = ''.join(['%5d' % x for x in BList[i:i + 4]])
                        f.write('CONECT%5d%s\n' % (a.Id, atmlist))
            f.write('END   \n')


class gro_file(IO_File):
    '''gro file'''

    def __init__(self, FILENAME):
        super().__init__(FILENAME)
        self.Mol = molecule(self.Proj)
        self.Comment = '\n'
        self.Box = [0.0 for _ in range(9)]

    def load(self, Scale2Ang=10):
        '''load molecule from file'''
        assert os.path.isfile(self.Name), "ERROR: Cannot find file %s!\n" % self.Name
        assert self.Ext.lower() in ('gro',), "ERROR: Wrong file format %s!\n" % self.Ext
        self.Lines = open(self.Name, 'rt').readlines()
        self.Comment = self.Lines[0]
        self.NAtoms = int(self.Lines[1])
        DataLines = [_ for _ in self.Lines[2:] if _.strip()]
        self.Box = [float(_) * Scale2Ang for _ in DataLines[-1].split()]
        NewChain = True
        NewRes = True
        for line in DataLines[:-1]:
            ResNo = int(line[:5])
            ResName = line[5:10].strip()
            name = line[10:15].strip()
            ele = name[0].upper()
            if ele not in Elements and len(name) > 1:
                ele = name[0].upper() + name[1].lower()
                if ele not in Elements:
                    raise RuntimeError('ERROR: Unkown element %s in atom %s (line: %s)' % (ele, name, line))
            AtomNo = int(line[15:20])
            x = float(line[20:28]) * Scale2Ang
            y = float(line[28:36]) * Scale2Ang
            z = float(line[36:44]) * Scale2Ang
            Atmp = atom(len(self.Mol.Atoms), name, atom_no(ele), np.array([x, y, z]), AtomNo)
            if not NewChain and (ResNo < self.Mol.Residues[-1].No):
                NewChain = True
            if not NewRes and (NewChain or ResNo != self.Mol.Residues[-1].No):
                NewRes = True
            if NewRes:
                Rtmp = residue(len(self.Mol.Residues), ResName, NO=ResNo)
                res = ResName[0].upper() + ResName[1:3].lower()
                if res in AA_List or res in NA_List or res[:2] in NA_List:
                    Rtmp.Type = res
                elif res[0] in 'NC':  # N or C terminal residues
                    res = ResName[:1].upper() + ResName[2:4].lower()
                    if res[1:] in AA_List:
                        Rtmp.Type = res
                if NewChain:
                    Ctmp = chain(len(self.Mol.Chains))
                    Ctmp.No = Ctmp.Id + 1
                    self.Mol.Chains.append(Ctmp)
                    NewChain = False
                else:
                    Ctmp = self.Mol.Chains[-1]
                Rtmp.ChainId = Ctmp.Id
                Rtmp.SeqNo = len(Ctmp.ResList)
                Ctmp.ResList.append(Rtmp.Id)
                self.Mol.Residues.append(Rtmp)
                NewRes = False
            else:
                Rtmp = self.Mol.Residues[-1]
            Atmp.ResId = Rtmp.Id
            Atmp.SeqNo = len(Rtmp.AtomList)
            Rtmp.AtomList.append(Atmp.Id)
            self.Mol.Atoms.append(Atmp)
        return self.Mol

    def save(self, Output=None, Mol=None, Scale2nm=0.1, Comments='', BoxList=[], WARNING=True):
        '''save molecule into file'''
        if not Output:
            Output = self.Name
        ext = get_ext(Output)
        assert ext.lower() in ('gro',), "ERROR: Wrong file format %s!\n" % ext
        if not Mol:
            Mol = self.Mol
        if WARNING and os.path.isfile(Output):
            print('WARNING: Overwriting existing file %s' % Output)
        if not Comments:
            Comments = self.Comment
        if Comments[-1] != '\n':
            Comments += '\n'
        SEARCH_MAX_XYZ = (not BoxList)
        if SEARCH_MAX_XYZ:
            BoxList = self.Box
            MaxXYZ = [0.0, 0.0, 0.0]
        with open(Output, 'wt') as f:
            f.write(Comments)
            f.write('%d\n' % (len(Mol.Atoms)))
            for c in Mol.Chains:
                for j in c.ResList:
                    res = Mol.Residues[j]
                    for k in res.AtomList:
                        atm = Mol.Atoms[k]
                        if SEARCH_MAX_XYZ:
                            for i in range(3):
                                MaxXYZ[i] = max(MaxXYZ[i], abs(atm.Coor[i]))
                        f.write('%5d%5s%5s%5d%8.3f%8.3f%8.3f\n' % (
                            res.No,
                            res.Name.upper(),
                            atm.Name.upper(),
                            atm.No,
                            atm.Coor[0] * Scale2nm,
                            atm.Coor[1] * Scale2nm,
                            atm.Coor[2] * Scale2nm,
                        ))

            f.write(''.join(['%10.5f' % (_ * Scale2nm) for _ in BoxList]) + '\n')


class DNA_sequence(object):
    """Sequence (5'->3') and name of DNA. Every 3 bases is represented by a 64-base number ('0-9A-Za-z@%')"""

    def __init__(self, N=0, BASE='', NAME='', SHORTNAME='', EVALFILE='', NAMESTR=''):
        self.NBase = N
        self.Base = BASE
        self.Name = NAME
        self.ShortName = SHORTNAME
        self.HOMO = -1
        self.EvalFile = EVALFILE
        self.Evals = []
        self.NameStr = NAMESTR
        self.Car = ''

    def __str__(self):
        s = 'Name : ' + self.Name + '\n'
        s += 'NBase: %4d' % self.NBase + '    ShortName: ' + self.ShortName + '\n'
        s += 'Base : ' + self.Base + '\n'
        return s

    def show(self):
        """Show infos of DNA"""
        print(self)

    def get_evals(self, Update=False):
        """Get evals of DNA"""
        if Update or not len(self.Evals):
            self.Evals = np.load(self.EvalFile)["arr_0"]
        return self.Evals

    def update_evals(self):
        """Update evals of DNA and return new DNA"""
        self.Evals = np.load(self.EvalFile)["arr_0"]
        return self

    @staticmethod
    def base2name(Base):
        """Convert base to name"""
        nums = [Base2num[b] for b in Base.upper()]
        N = len(nums)
        if N % 3:
            nums.extend([0] * (3 - N % 3))
        nums64 = [nums[i] * 16 + nums[i + 1] * 4 + nums[i + 2] for i in range(0, N, 3)]
        return ''.join([Base64[_] for _ in nums64])

    @staticmethod
    def name2base(Name, N):
        """Convert name to base"""
        nums64 = [Base64_Dict[_] for _ in Name]
        base = ''
        for n in nums64:
            base += Base_List[(n & 48) >> 4] + Base_List[(n & 12) >> 2] + Base_List[n & 3]
        return base[:N]

    @classmethod
    def get_from_base(cls, BASE, SHORTNAME='', EVALFILE=''):
        """Get a new DNA from bases"""
        return cls(len(BASE), BASE, cls.base2name(BASE), SHORTNAME, EVALFILE)

    @classmethod
    def get_from_name(cls, NAME, N, SHORTNAME='', EVALFILE=''):
        """Get a new DNA from 64-base name"""
        return cls(N, cls.name2base(NAME, N), NAME, SHORTNAME, EVALFILE)


class RNA_sequence(object):
    """Sequence (5'->3') and name of DNA. Every 3 bases is represented by a 64-base number ('0-9A-Za-z@%')"""

    def __init__(self, N=0, BASE='', NAME='', SHORTNAME='', EVALFILE=''):
        self.NBase = N
        self.Base = BASE
        self.Name = NAME
        self.ShortName = SHORTNAME
        self.HOMO = -1
        self.EvalFile = EVALFILE
        self.Evals = []
        self.Car = ''

    def __str__(self):
        s = 'Name : ' + self.Name + '\n'
        s += 'NBase: %4d' % self.NBase + '    ShortName: ' + self.ShortName + '\n'
        s += 'Base : ' + self.Base + '\n'
        return s

    def show(self):
        """Show infos of DNA"""
        print(self)

    def get_evals(self, Update=False):
        """Get evals of DNA"""
        if Update or not len(self.Evals):
            self.Evals = np.load(self.EvalFile)["arr_0"]
        return self.Evals

    def update_evals(self):
        """Update evals of DNA and return new DNA"""
        self.Evals = np.load(self.EvalFile)["arr_0"]
        return self

    @staticmethod
    def base2name(Base):
        """Convert base to name"""
        nums = [RNABase2num[b] for b in Base.upper()]
        N = len(nums)
        if N % 3:
            nums.extend([0] * (3 - N % 3))
        nums64 = [nums[i] * 16 + nums[i + 1] * 4 + nums[i + 2] for i in range(0, N, 3)]
        return ''.join([Base64[_] for _ in nums64])

    @staticmethod
    def name2base(Name, N):
        """Convert name to base"""
        nums64 = [Base64_Dict[_] for _ in Name]
        base = ''
        for n in nums64:
            base += RNABase_List[(n & 48) >> 4] + RNABase_List[(n & 12) >> 2] + RNABase_List[n & 3]
        return base[:N]

    @classmethod
    def get_from_base(cls, BASE, SHORTNAME='', EVALFILE=''):
        """Get a new DNA from bases"""
        return cls(len(BASE), BASE, cls.base2name(BASE), SHORTNAME, EVALFILE)

    @classmethod
    def get_from_name(cls, NAME, N, SHORTNAME='', EVALFILE=''):
        """Get a new DNA from 64-base name"""
        return cls(N, cls.name2base(NAME, N), NAME, SHORTNAME, EVALFILE)

def rand_seq(Len, NA):
    """Random sequence of DNA with length Len"""
    return ''.join([NA[_] for _ in np.random.randint(0, len(NA), Len)])


def reverse_seq(s):
    return ''.join([DNA_pair[_] for _ in s[::-1]])

def reverse_RNA_seq(s):
    return ''.join([RNA_pair[_] for _ in s[::-1]])


def read_DNA_info(File, EvalsDB=None):
    """Read DNA sequence, name, evals_filename from File, and evals from EvalsDB (if provided)"""
    if not os.path.isfile(File):
        print('WARNING: Fail to find file %s !' % File)
        return []
    NameStrs = read_txt_lines(File, SPLIT=True)
    DNAs = []
    if EvalsDB is not None and os.path.isfile(EvalsDB):
        Data = load_fsz_thread(EvalsDB, AlwaysReturnDict=True)
    else:
        Data = {}
    for x in NameStrs:
        s = x[0].split('_')
        d = DNA_sequence.get_from_name(s[2], int(s[1]), x[-1], x[0] + '.npz')
        d.HOMO = int(x[1])
        if d.EvalFile in Data.keys():
            d.Evals = Data[d.EvalFile]
        d.NameStr = '\t'.join(x)
        DNAs.append(d)
    return DNAs

def read_RNA_info(File, EvalsDB=None):
    """Read RNA sequence, name, evals_filename from File, and evals from EvalsDB (if provided)"""
    if not os.path.isfile(File):
        print('WARNING: Fail to find file %s !' % File)
        return []
    NameStrs = read_txt_lines(File, SPLIT=True)
    RNAs = []
    if EvalsDB is not None and os.path.isfile(EvalsDB):
        Data = load_fsz_thread(EvalsDB, AlwaysReturnDict=True)
    else:
        Data = {}
    for x in NameStrs:
        s = x[0].split('_')
        d = RNA_sequence.get_from_name(s[2], int(s[1]), x[-1], x[0] + '.npz')
        d.HOMO = int(x[1])
        if d.EvalFile in Data.keys():
            d.Evals = Data[d.EvalFile]
        RNAs.append(d)
    return RNAs

def find_DNA(Tag_DNA, DNAs):
    """Search for single DNA from known DNAs, return the DNA if found."""
    for d in DNAs:
        if Tag_DNA in (d.Name, d.ShortName, d.Base):
            return d
    else:
        raise AssertionError("ERROR: Fail to find DNA %s!" % Tag_DNA)


def find_DNA_no(Tag_DNA, DNAs):
    """Search for single DNA from known DNAs, return the No. of Tag_DNA if found"""
    for i, d in enumerate(DNAs):
        if Tag_DNA in (d.Name, d.ShortName, d.Base):
            return i
    else:
        return -1


def dopant_of_DNA(MainBase):
    return ''.join([b for b in 'AGCT' if b != MainBase])


def get_dopant(Seq, MainBase='C'):
    """Return dopant of DNA"""
    Dope_Nst = Dope_Ned = -1
    for i, s in enumerate(Seq):
        if s != MainBase:
            if Dope_Nst < 0:
                Dope_Nst = i
                Dope_Ned = i
            else:
                Dope_Ned = max(i, Dope_Ned)
    return Seq[Dope_Nst:Dope_Ned + 1]


def anti_seq(Seq, DNA=True):
    """Anti-sequence of NA"""
    pairdict = DNA_pair if DNA else RNA_pair
    return ''.join([pairdict[b] for b in Seq[::-1]])


def unique_DNA_seqs(SeqList):
    """Unique DNA sequences"""
    UnqSeqs = set(SeqList)
    return [s for i, s in enumerate(UnqSeqs) if anti_seq(s) not in UnqSeqs[:i]]


# Banned_Seqs = ['GAATA','TATTC','AGAGA','TGAGA','AGAGC','GAC']
# Banned_Seqs = ['AGATA', 'AGACA', 'AGAGA', 'GAATA']
# Banned_Seqs = set(Banned_Seqs + [anti_seq(s) for s in Banned_Seqs])
Banned_Seqs = []

# def allow_cut(Seq, CutPos=0, MainBase='C'):
#     """Whether Seq can be cut at CutPos"""
#     if Seq[CutPos] in ['A', 'T']:
#         return True
#     elif Seq[CutPos] == 'C':
#         return False
#     else:
#         if CutPos < 0:
#             CutPos += len(Seq)
#         if len(Seq) > CutPos + 1 and Seq[CutPos + 1] != 'C':
#             return False
#         elif MainBase != 'C':
#             return False
#         if CutPos > 0 and Seq[CutPos - 1] != 'C':
#             return False
#         elif MainBase != 'C':
#             return False
#         return True


def allow_cut(Seq, CutPos=0):
    """Whether Seq can be cut at CutPos"""
    # if Seq[CutPos] == 'T':
    #     if (len(Seq) > CutPos + 1 and Seq[CutPos + 1] =='C') or (CutPos > 0 and Seq[CutPos - 1]  == 'C'):
    #         return False
    #     else:
    #         return True
    # elif Seq[CutPos] == 'A':
    #     if (len(Seq) > CutPos + 1 and Seq[CutPos + 1] =='G') or (CutPos > 0 and Seq[CutPos - 1]  == 'G'):
    #         return False
    #     else:
    #         return True
    if Seq[CutPos] in ('A', 'T'):
        return True
    elif Seq[CutPos] == 'C':
        return False
        # if CutPos < 0:
        #     CutPos += len(Seq)
        # if len(Seq) > CutPos + 1 and Seq[CutPos + 1] != 'G':
        #     return False
        # if CutPos > 0 and Seq[CutPos - 1] != 'G':
        #     return False
        # return True
    else:
        return False
        # if CutPos < 0:
        #     CutPos += len(Seq)
        # if len(Seq) > CutPos + 1 and Seq[CutPos + 1] != 'C':
        #     return False
        # # elif MainBase != 'C':
        # #     return False
        # if CutPos > 0 and Seq[CutPos - 1] != 'C':
        #     return False
        # # elif MainBase != 'C':
        # #     return False
        # return True


def cut_bms(Seq, DEBUG=False):
    """Cut DNA into basemodes"""
    LSeq = len(Seq)
    if LSeq <= 3:
        return [Seq]
    NotBanned = [-1 for _ in range(LSeq)]
    for bs in Banned_Seqs:
        if bs in Seq:
            Lbs = len(bs)
            for p in find_all(Seq, bs, Overlap=True):
                NotBanned[p] = 1 if bs[0] in ('A', 'T') and NotBanned[p] != 0 else 0
                for i in range(1, Lbs - 1):
                    NotBanned[p + i] = 0
                pp = p + Lbs - 1
                NotBanned[pp] = 1 if bs[-1] in ('A', 'T') and NotBanned[pp] != 0 else 0
    secs = [[0]]
    cuttable = []
    cutinfo = []
    for i, notban in enumerate(NotBanned):
        if notban < 0:
            MyCut = allow_cut(Seq, i)
        else:
            MyCut = True if notban > 0 else False
        cutinfo.append(MyCut)
        if DEBUG:
            print(i, MyCut)
        if i == 0:
            if not MyCut:
                cuttable.append(MyCut)
            continue
        idx_head = secs[-1][0]
        if len(cuttable) < len(secs):
            if not cutinfo[idx_head]:
                cuttable.append(False)
            else:
                cuttable.append(MyCut)
        if MyCut:
            if not cuttable[-1]:
                secs[-1].append(i)
                if i != LSeq - 1:
                    secs.append([i])
        else:
            if cutinfo[i - 1] and idx_head != i - 1:
                secs[-1].append(i - 1)
                secs.append([i - 1])
                cuttable.append(False)
    if len(secs[-1]) < 2:
        secs[-1].append(LSeq - 1)
    if DEBUG:
        print(secs)
        print(cuttable)
    bms = []
    for c, idx in zip(cuttable, secs):
        if not c:
            bms.append(Seq[idx[0]:idx[1] + 1])
        else:
            DopantSec = Seq[idx[0]:idx[1] + 1]
            LSec = len(DopantSec)
            N = math.ceil((LSec - 1) / 3)
            Leach = LSec / N
            bms.extend([DopantSec[int(i * Leach):int((i + 1) * Leach) + 1] for i in range(N)])
    return bms


def cut_bms2(Seq, MULTI=False, DEBUG=False):
    """Cut DNA into basemodes"""
    LSeq = len(Seq)
    if LSeq <= 3:
        return [Seq]
    NotBanned = [-1 for _ in range(LSeq)]
    for bs in Banned_Seqs:
        if bs in Seq:
            Lbs = len(bs)
            for p in find_all(Seq, bs, Overlap=True):
                NotBanned[p] = 1 if bs[0] in ('A', 'T') and NotBanned[p] != 0 else 0
                for i in range(1, Lbs):
                    NotBanned[p + i] = 0
                # for i in range(1, Lbs - 1):
                #     NotBanned[p + i] = 0
                pp = p + Lbs
                if pp < LSeq:
                    NotBanned[pp] = 2 if bs[-1] in ('A', 'T') else 0
    secs = [[0]]
    cutmark = [True if NotBanned[i] > 1 or (NotBanned[i] and b in 'AT') else False for i, b in enumerate(Seq)]
    cutinfo = cutmark
    # cutinfo = [True] + [(a and b) for a, b in zip(cutmark[:-1], cutmark[1:])]
    cuttable = []
    for i, MyCut in enumerate(cutinfo):
        if DEBUG:
            print(i, MyCut)
        if i == 0:
            if not MyCut:
                cuttable.append(MyCut)
            continue
        idx_head = secs[-1][0]
        if len(cuttable) < len(secs):
            if not cutinfo[idx_head]:
                cuttable.append(False)
            else:
                cuttable.append(MyCut)
        if MyCut:
            if not cuttable[-1]:
                secs[-1].append(i)
                if i != LSeq - 1:
                    secs.append([i])
        else:
            if cutinfo[i - 1] and idx_head != i - 1:
                secs[-1].append(i - 1)
                secs.append([i - 1])
                cuttable.append(False)
    if len(secs[-1]) < 2:
        secs[-1].append(LSeq)
    if DEBUG:
        print(secs)
        print(cuttable)
    LSecs = len(secs)
    RelocateInfo = []
    for i, (c, idx) in enumerate(zip(cuttable, secs)):
        if not c and Seq[idx[1] - 1] in 'CG':  # idx[1] - idx[0] <= 2:
            if i == LSecs - 1:
                pass
                # if not cutinfo[idx[0]] and cuttable[-2]:
                #     RelocateInfo.append((i, -1))
            elif i == 0:
                if LSecs > 0 and cuttable[1]:
                    RelocateInfo.append(0)
            else:
                if cuttable[i + 1]:
                    RelocateInfo.append(i)
    if RelocateInfo:
        for i in RelocateInfo[::-1]:
            secs[i + 1][0] += 1
            secs[i][1] += 1
            if secs[i + 1][1] == secs[i + 1][0]:
                del cuttable[i + 1]
                del secs[i + 1]
    # LSecs = len(secs)
    # ShrinkInfo = []
    # for i, (c, idx) in enumerate(zip(cuttable, secs)):
    #     if c and idx[1] - idx[0] == 1:
    #         if i == 0:
    #             ShrinkInfo.append((0, 1))
    #         elif i == LSecs - 1:
    #             ShrinkInfo.append((i, -1))
    #         else:
    #             LLeft = secs[i - 1][1] - secs[i - 1][0]
    #             LRight = secs[i + 1][1] - secs[i + 1][0]
    #             if LLeft < LRight:
    #                 ShrinkInfo.append((i, -1))
    #             else:
    #                 ShrinkInfo.append((i, 1))
    # if ShrinkInfo:
    #     for i, p in ShrinkInfo[::-1]:
    #         if p < 0:
    #             secs[i - 1][1] = secs[i][1]
    #         else:
    #             secs[i + 1][0] = secs[i][0]
    #         del cuttable[i]
    #         del secs[i]
    if DEBUG:
        print(secs)
        print(cuttable)
    if MULTI:
        bms_all = []
        for l in (2, 3, 4):
            bms = []
            for c, idx in zip(cuttable, secs):
                if not c:
                    bms.append(Seq[idx[0]:idx[1]])
                else:
                    DopantSec = Seq[idx[0]:idx[1]]
                    LSec = len(DopantSec)
                    N = math.ceil(LSec / l)
                    Leach = LSec / N
                    bms.extend([DopantSec[int(i * Leach):int((i + 1) * Leach)] for i in range(N)])
            if bms not in bms_all:
                bms_all.append(bms)
        return bms_all
    else:
        bms = []
        for c, idx in zip(cuttable, secs):
            if not c:
                bms.append(Seq[idx[0]:idx[1]])
            else:
                DopantSec = Seq[idx[0]:idx[1]]
                LSec = len(DopantSec)
                N = math.ceil((LSec) / 4)
                Leach = LSec / N
                bms.extend([DopantSec[int(i * Leach):int((i + 1) * Leach)] for i in range(N)])
        return bms


def cut_multi_bms(Seq, DEBUG=False):
    """Cut DNA into basemodes"""
    LSeq = len(Seq)
    if LSeq <= 3:
        return [Seq]
    NotBanned = [-1 for _ in range(LSeq)]
    for bs in Banned_Seqs:
        if bs in Seq:
            Lbs = len(bs)
            for p in find_all(Seq, bs, Overlap=True):
                NotBanned[p] = 1 if bs[0] in ('A', 'T') and NotBanned[p] != 0 else 0
                for i in range(1, Lbs - 1):
                    NotBanned[p + i] = 0
                pp = p + Lbs - 1
                NotBanned[pp] = 1 if bs[-1] in ('A', 'T') and NotBanned[pp] != 0 else 0
    secs = [[0]]
    cuttable = []
    cutinfo = []
    for i, notban in enumerate(NotBanned):
        if notban < 0:
            MyCut = allow_cut(Seq, i)
        else:
            MyCut = True if notban > 0 else False
        cutinfo.append(MyCut)
        if DEBUG:
            print(i, MyCut)
        if i == 0:
            if not MyCut:
                cuttable.append(MyCut)
            continue
        idx_head = secs[-1][0]
        if len(cuttable) < len(secs):
            if not cutinfo[idx_head]:
                cuttable.append(False)
            else:
                cuttable.append(MyCut)
        if MyCut:
            if not cuttable[-1]:
                secs[-1].append(i)
                if i != LSeq - 1:
                    secs.append([i])
        else:
            if cutinfo[i - 1] and idx_head != i - 1:
                secs[-1].append(i - 1)
                secs.append([i - 1])
                cuttable.append(False)
    if len(secs[-1]) < 2:
        secs[-1].append(LSeq - 1)
    if DEBUG:
        print(secs)
        print(cuttable)
    bms_all = []
    for l in (1, 2, 3):
        bms = []
        for c, idx in zip(cuttable, secs):
            if not c:
                bms.append(Seq[idx[0]:idx[1] + 1])
            else:
                DopantSec = Seq[idx[0]:idx[1] + 1]
                LSec = len(DopantSec)
                N = math.ceil((LSec - 1) / l)
                Leach = LSec / N
                bms.extend([DopantSec[int(i * Leach):int((i + 1) * Leach) + 1] for i in range(N)])
        if bms not in bms_all:
            bms_all.append(bms)
    return bms_all


def dopant_to_seq(Dopant, MainBase='C', TotalLen=50):
    """Generate DNA sequence from dopant"""
    LDopant = len(Dopant)
    LHead = (TotalLen - LDopant) // 2
    LTail = TotalLen - LDopant - LHead
    return MainBase * LHead + Dopant + MainBase * LTail


def dopant_to_name(dopant, MainBase='C', TotalLen=50):
    """Get DNA name from dopant"""
    if len(dopant) == 0:
        return MainBase + '%d' % TotalLen
    elif dopant[0] == MainBase or dopant[-1] == MainBase:
        Dope_Nst = Dope_Ned = -1
        for i, s in enumerate(dopant):
            if s != MainBase:
                if Dope_Nst < 0:
                    Dope_Nst = i
                else:
                    Dope_Ned = max(i, Dope_Ned)
        if Dope_Nst < 0:
            return MainBase + '%d' % TotalLen
        else:
            Dope_Ned = max(Dope_Nst, Dope_Ned) + 1
            return dopant_to_name(dopant[Dope_Nst:Dope_Ned], MainBase, TotalLen)
    else:
        NDope = len(dopant)
        NRest = TotalLen - NDope
        if NDope == 1:
            return MainBase + '%d' % (NRest) + dopant + '1'
        NInner = dopant.count(MainBase)
        if NInner == NDope - 2:
            HTName = dopant[0] + ('2' if dopant[0] == dopant[-1] else dopant[-1])
            Name = MainBase + '%d' % (NRest + NInner) + HTName + '-%d' % NInner
        else:
            Head = MainBase + '%d' % (NRest)
            if NDope >= 5:
                if NDope > 5:
                    LHead = NRest // 2
                    LTail = NRest - LHead
                    if LHead == LTail:
                        Head = MainBase + '%dx2' % (LHead)
                    else:
                        Head = MainBase + '%d-%d' % (LHead, LTail)
                Name = Head + '-' + shrink_DNA_name(dopant, ShrinkLen=5, CompleteSeq=False)
            else:
                UnqBase = np.unique(list(dopant))
                if NDope == 4:
                    UnqBase2 = np.unique(list(dopant[1:-1]))
                    if len(UnqBase) == 1:
                        Name = Head + dopant[0] + '%d' % (NDope) + '-0'
                    else:
                        if len(UnqBase2) == 1 and dopant[0] != dopant[1] and dopant[-1] != dopant[1]:
                            HTName = dopant[0] + ('2' if dopant[0] == dopant[-1] else dopant[-1])
                            Name = Head + HTName + '-2' + dopant[1]
                        else:
                            Name = Head + '-' + dopant
                elif NDope == 3:
                    if len(UnqBase) == 1:
                        Name = Head + dopant[0] + '%d' % (NDope) + '-0'
                    elif dopant[0] == dopant[-1] and dopant[0] != dopant[1]:
                        Name = Head + dopant[0] + '2-1' + dopant[1]
                    else:
                        Name = Head + '-' + dopant
                else:
                    print('ERROR')
    return Name


def seq_to_basecount(Sequence):
    """Count adjacent base and return as a list"""
    BaseCount = [[Sequence[0], 1]]
    for s in Sequence[1:]:
        if s == BaseCount[-1][0]:
            BaseCount[-1][1] += 1
        else:
            BaseCount.append([s, 1])
    return BaseCount


def basecount_to_name(BaseCount, ShrinkLen=4):
    """Generate name from BaseCount list"""
    name = []
    Case = -1
    for b, n in BaseCount:
        if n >= ShrinkLen:
            name.append('-%s%d' % (b, n))
            Case = 0
        else:
            if Case == 0:
                name.append('-')
            name.extend([b for _ in range(n)])
            Case = 1
    name = ''.join(name)
    if name[0] == '-':
        name = name[1:]
    return name


def shrink_DNA_name(Sequence, ShrinkLen=4, CompleteSeq=True):
    """Shrinked DNA name"""
    BaseCount = seq_to_basecount(Sequence)
    if CompleteSeq and len(BaseCount) > 1 and Sequence[0] == Sequence[-1]:
        MainBase, LHead = BaseCount[0]
        LTail = BaseCount[-1][1]
        if LHead == LTail:
            Head = MainBase + '%dx2' % (LHead)
        else:
            Head = MainBase + '%d-%d' % (LHead, LTail)
        return Head + '-' + basecount_to_name(BaseCount[1:-1], ShrinkLen=ShrinkLen)
    else:
        return basecount_to_name(BaseCount, ShrinkLen=ShrinkLen)


def seq_to_shortname(Sequence):
    """Generate short name for DNA"""
    LSeq = len(Sequence)
    if Sequence[0] == Sequence[-1]:
        MainBase = Sequence[0]
        Dopant = get_dopant(Sequence, MainBase)
        if Sequence == dopant_to_seq(Dopant, MainBase, LSeq):
            return dopant_to_name(Dopant, MainBase, LSeq)
    return shrink_DNA_name(Sequence)


def mol_NA(Sequence, Type, Topo='L'):
    """Generate a molecule of DNA/RNA"""
    Type = Type.strip().lower()
    assert Type in helix_dict
    Helix, HHeight, HRotation = helix_dict[Type]
    IS_DNA = (Type[-3:] == 'dna')
    if IS_DNA:
        NA_pair = DNA_pair
        WrongBase = {'U': 'T'}
        base2name = DNA_base2name
        mol = molecule(TYPE='DSDNA')
    else:
        NA_pair = RNA_pair
        WrongBase = {'T': 'U'}
        base2name = RNA_base2name
        mol = molecule(TYPE='DSRNA')
    Sequence = list(Sequence.strip().upper())
    AntiSeq = []
    for i, x in enumerate(Sequence):
        assert x in NA_base
        if x in WrongBase:
            Sequence[i] = x = WrongBase[x]
        AntiSeq += NA_pair[x]
    mol.Chains = [chain(0, 'A'), chain(1, 'B')]
    Nseq = len(Sequence)
    if Topo == 'R':
        RingR = Nseq * HHeight / (2 * math.pi)
        z2phi = 1 / RingR
    for c in mol.Chains:
        if c.Id == 1:
            seq = AntiSeq
            # theta_shift=0
            zdir = 1
        else:
            seq = Sequence
            # theta_shift=180
            zdir = -1
        for i, s in enumerate(seq):
            MyHeight = i * HHeight
            MyRotation = i * HRotation
            if c.Id == 0:
                rid = i
                sqno = i
            else:
                rid = 2 * Nseq - 1 - i
                sqno = Nseq - 1 - i
            rsd = residue(rid, base2name[s], TYPE=s, NO=rid + 1, CHAINID=c.Id, SEQNO=sqno)
            if c.Id == 0:
                c.ResList.append(rsd.Id)
                mol.Residues.append(rsd)
            else:
                c.ResList.insert(0, rsd.Id)
                mol.Residues.insert(Nseq, rsd)
            if 0 < i < Nseq - 1 or Topo == 'R':
                Not_Shown = ["HO3'", "HO5'"]
            elif i == 0:
                if c.Id == 0:
                    Not_Shown = ["HO3'", "P", "OP1", "OP2"]
                else:
                    Not_Shown = ["HO5'"]
            else:
                if c.Id == 0:
                    Not_Shown = ["HO5'"]
                else:
                    Not_Shown = ["HO3'", "P", "OP1", "OP2"]
            for record in Helix[s]:
                name, r, theta, z = record
                if name in Not_Shown:
                    continue
                ele = sep_num_abc(name)[0]
                ele = ele[0].upper() + ele[1].lower() if len(ele) > 1 else ele[0].upper()
                if ele not in Elements[:54] and ele[0] in Elements:
                    ele = ele[0]
                xyz = np.zeros((3,))
                theta = (zdir * theta + MyRotation) * deg2rad
                if Topo == 'R':
                    phi = (z * zdir + MyHeight) * z2phi
                    newr = RingR + r * math.sin(theta)
                    xyz[0] = newr * math.cos(phi)
                    xyz[1] = newr * math.sin(phi)
                    xyz[2] = r * math.cos(theta)
                else:
                    xyz[0] = r * math.cos(theta)
                    xyz[1] = r * math.sin(theta)
                    xyz[2] = z * zdir + MyHeight
                atm = atom(len(mol.Atoms), name, atom_no(ele), xyz, RESID=rsd.Id, SEQNO=len(rsd.AtomList))
                rsd.AtomList.append(atm.Id)
                mol.Atoms.append(atm)
    return mol


class basemode_coef(object):

    def __init__(self, NAME='', BASEMODES=[], COEFS=[]):
        self.Name = NAME
        self.BaseModes = BASEMODES
        self.Coefs = COEFS

    def __eq__(self, other):
        assert isinstance(other, basemode_coef), "ERROR: Data type not match!\n"
        if self.Name == other.Name:
            return (np.all(self.BaseModes == other.BaseModes) and np.all(self.Coefs == other.Coefs))
        else:
            return False

    def __ne__(self, other):
        assert isinstance(other, basemode_coef), "ERROR: Data type not match!\n"
        return not (self == other)

    def get_lines(self, StrList=False):
        lines = [self.Name + ':', ' '.join(['%10s' % bm for bm in self.BaseModes]), ' '.join(['%10.6f' % k for k in self.Coefs])]
        if StrList:
            return [_ + '\n' for _ in lines]
        else:
            return '\n'.join(lines)

    @classmethod
    def get_from_lines(cls, NAMELINE, BMLINE, COEFLINE):
        return cls(NAME=NAMELINE[:-1], BASEMODES=BMLINE.split(), COEFS=[float(x) for x in COEFLINE.split()])


def save_NA(Name, Seq, NAType, SaveXyz=False):
    if Name[-4:] != '.pdb':
        Name = Name + '.pdb'
    mol = mol_NA(Seq, NAType)
    pdb_file(Name).save(Mol=mol, BOND=False)
    if SaveXyz:
        XyzData = []
        for c in mol.Chains:
            for j in c.ResList:
                res = mol.Residues[j]
                for k in res.AtomList:
                    atm = mol.Atoms[k]
                    XyzData.append([atm.Id, atm.Elem, 0, atm.Coor[0], atm.Coor[1], atm.Coor[2]])
        # for atm in mol.Atoms:
        #     XyzData.append([atm.Id, atm.Elem, 0, atm.Coor[0], atm.Coor[1], atm.Coor[2]])
        save_xyz(Name + '.xyz', np.array(XyzData))

def save_DNA(*args, **kwargs):
    save_NA(*args, **kwargs)

InfoFormat = ('%s_Info.txt', 'infos' + os.sep + '%s_Info.txt')
NameFileFormat = {
    'P': ('Name', '%s.pdb'),
    'E': ('FullName', '%s.npz'),
    # 'C': ('Name', '%s_total.xyz'),
    'C': ('NamePart', '%s_Part_0-%s.csv', 'infos' + os.sep + '%s_Part_0-%s.csv'),
    'H': ('NamePart', '%s_H_0-%s.npz', '%s_H_0-%s.fsz'),
    'S': ('NamePart', '%s_S_0-%s.npz', '%s_S_0-%s.fsz'),
    'M': ('FullNameSk', 'Evcts_%s%s.npz', 'Evcts_%s%s.fsz'),
    'CB': ('NameSk', 'CB_%s%s.npz', 'CB_%s%s.fsz'),
}


class DNAProj(DNA_sequence):
    """Informations of DNA"""

    def __init__(self, N=0, BASE='', BASENAME='', FULLNAME='', SHORTNAME='', EVALFILE=''):
        super().__init__(N=N, BASE=BASE, NAME=FULLNAME, SHORTNAME=SHORTNAME, EVALFILE=EVALFILE)
        self.FullName = FULLNAME
        self.BaseName = BASENAME
        self.SkMO = (-1, -1)
        self.SkStr = ''
        self.FileNames = {}

    @classmethod
    def get_from_line(cls, Line):
        """Get a new DNAProj from str line"""
        NameStr = Line.split()
        s = NameStr[0].split('_')
        n = int(s[1])
        p = cls(BASENAME=s[2], N=n, BASE=cls.name2base(s[2], n), FULLNAME=NameStr[0], SHORTNAME=NameStr[-1], EVALFILE=NameStr[0] + '.npz')
        p.HOMO = int(NameStr[1])
        if len(NameStr) > 3:
            p.SkMO = tuple([int(_) for _ in NameStr[2].split(',')])
            p.SkStr = '_sk%d-%d' % p.SkMO
        return p

    def search_files(self, NPart=None):
        """Search FileNames of DNA"""
        if NPart is None:
            for fmt in InfoFormat:
                InfoName = fmt % self.BaseName
                if os.path.isfile(InfoName):
                    break
            else:
                raise FileNotFoundError('ERROR: Fail to find info file!')
            NPart = info_file(InfoName).load().NPart
        self.SkStr = '_sk%d-%d' % self.SkMO if self.SkMO[0] >= 0 else ''
        NameDict = {
            'Name': self.BaseName,
            'FullName': self.FullName,
            'NamePart': (self.BaseName, NPart - 1),
            'FullNameSk': (self.FullName, self.SkStr),
            'NameSk': (self.BaseName, self.SkStr),
        }
        for k, v in NameFileFormat.items():
            MyName = NameDict[v[0]]
            for fmt in v[1:]:
                FileName = fmt % MyName
                if os.path.isfile(FileName):
                    break
            else:
                print('WARNING: Fail to find %s file! (%s)' % (k, FileName))
            self.FileNames[k] = FileName
        return self.FileNames


class SecNode(object):
    """Node of SecGraph"""

    def __init__(self, ID, SEC):
        self.Id = ID
        self.Name = SEC
        self.Head = SEC[0]
        self.Tail = SEC[-1]
        self.Parents = []
        self.Children = []
        self.Visited = False


class SecGraph(object):
    """Graph made by DNA sections"""

    def __init__(self, SECS, HEAD, TAIL):
        self.NSec = len(SECS)
        self.Nodes = [SecNode(0, 'S' + HEAD), SecNode(1, TAIL + 'E')] + [SecNode(i + 2, _) for i, _ in enumerate(SECS)]
        self.NNode = len(self.Nodes)
        for i in range(self.NNode):
            nd = self.Nodes[i]
            for j in range(i):
                nd2 = self.Nodes[j]
                if nd2.Head == nd.Tail:
                    nd.Children.append(j)
                    nd2.Parents.append(i)
                if nd.Head == nd2.Tail:
                    nd2.Children.append(i)
                    nd.Parents.append(j)

    def sort_all_children(self):
        for i, nd in enumerate(self.Nodes):
            self.Nodes[i].Visited = False
            if len(nd.Children) <= 1:
                continue
            SortList = []
            for c in nd.Children:
                nd2 = self.Nodes[c]
                if nd2.Name == nd.Name:
                    t = 2
                elif nd2.Head + nd2.Tail == nd.Head + nd.Tail:
                    t = 1
                else:
                    t = 0
                SortList.append((len(nd2.Name), t))
            NewList = [nd.Children[_] for _ in np.lexsort(np.array(SortList).T.tolist())]
            self.Nodes[i].Children = NewList

    def search_path(self, REVERSE=False):
        self.sort_all_children()
        Path = []
        Stack = [0]
        while len(Stack):
            nd = self.Nodes[Stack[-1]]
            for no in nd.Children:
                if not self.Nodes[no].Visited:
                    Stack.append(no)
                    self.Nodes[no].Visited = True
                    break
            else:
                Path.append(Stack.pop())
        if len(Path) == self.NNode:
            GoodPath = [self.Nodes[_].Name for _ in Path[self.NNode - 2:0:-1]]
            # print('Good Path:', GoodPath)
            return GoodPath
        else:
            if REVERSE:
                NewSecs = [self.Nodes[_].Name for _ in Path if _ not in (0, 1)]
                NewSecs.extend([reverse_seq(self.Nodes[_].Name) for _ in Stack if _ not in (0, 1)])
                tmp = SecGraph(NewSecs, self.Nodes[0].Tail, self.Nodes[1].Head)
                return tmp.search_path()
            else:
                print('WRONG Path:', Path)
                print(' '.join([self.Nodes[_].Name for _ in Path]))
                self.show()
                raise RuntimeError('Fail to find a good path for secs:\n%s' % (','.join([_.Name for _ in self.Nodes[2:]])))

    def show(self):
        print('SecGraph Info:')
        for nd in self.Nodes:
            print('%3d %s: %s' % (nd.Id, nd.Name, ' '.join(['%2d' % _ for _ in nd.Children])))


def sec2seq(coefs, DEBUG=False, REVERSE=False):
    if DEBUG:
        print('DEBUG INFORMATION:\n', coefs)
    secs = []
    nums = []
    HeadCount = {_: [] for _ in 'AGCT'}
    TailCount = {_: [] for _ in 'AGCT'}
    HeadNum = {_: 0 for _ in 'AGCT'}
    TailNum = {_: 0 for _ in 'AGCT'}
    for i, (s, n) in enumerate(coefs):
        nums.append(n)
        secs.append(s)
        HeadCount[s[0]].append(i)
        HeadNum[s[0]] += n
        TailCount[s[-1]].append(i)
        TailNum[s[-1]] += n
    NSecs = np.sum(nums)
    if NSecs == 1:
        return secs[0]
    # SecSequence = [-1 for _ in range(NSecs)]
    # HeadPos = {_: 0 for _ in 'AGCT'}
    # TailPos = {_: 0 for _ in 'AGCT'}
    TestHeadTail = np.array([HeadNum[_] - TailNum[_] for _ in 'AGCT'])
    if np.sum(TestHeadTail) != 0 or np.any(np.abs(TestHeadTail) > 1):
        print('TestHeadTail:', TestHeadTail)
        print('H: ', HeadCount, 'T: ', TailCount)
        raise RuntimeError('ERROR: Wrong combination of sections:\n' + str(coefs))
    if HeadNum['C'] + HeadNum['G'] > 1 or TailNum['C'] + TailNum['G'] > 1:
        raise RuntimeError('ERROR: Too many sections starting(ending) by "C" and "G":\n' + str(coefs))
    if HeadCount['C']:
        HeadBase = 'C'
    elif HeadCount['G']:
        HeadBase = 'G'
    else:
        for b in 'AGCT':
            if HeadNum[b] - TailNum[b] == 1:
                HeadBase = b
                break
        else:
            HeadBase = ''
    if TailCount['C']:
        TailBase = 'C'
    elif TailCount['G']:
        TailBase = 'G'
    else:
        for b in 'AGCT':
            if HeadNum[b] - TailNum[b] == -1:
                TailBase = b
                break
        else:
            TailBase = ''
    for b in 'AGCT':
        delta = HeadNum[b] - TailNum[b]
        if b == HeadBase:
            delta -= 1
        if b == TailBase:
            delta += 1
        if delta != 0:
            print('H: ', HeadBase, 'T: ', TailBase)
            print('Bad H-T number for base %s: %d' % (b, delta))
            print('TestHeadTail:', TestHeadTail)
            print('H: ', HeadCount, 'T: ', TailCount)
            raise RuntimeError('ERROR: Wrong combination of sections:\n' + str(coefs))
    if not (HeadBase or TailBase):
        for k, v in HeadCount.items():
            if v:
                HeadBase = k
                break
    # HEAD_START = True
    # if TailBase and not HeadBase:
    #     HEAD_START = False
    # # NSecLeft = NSecs
    # myHead, myTail = 0, NSecs - 1
    if DEBUG:
        print('H: ', HeadBase, 'T: ', TailBase)
        print('H: ', HeadCount, 'T: ', TailCount)
        print(nums)
    # HeadSecNum = {_: len(HeadCount[_]) for _ in 'AGCT'}
    # TailSecNum = {_: len(TailCount[_]) for _ in 'AGCT'}
    # if HeadBase and HeadNum[HeadBase] == 1:
    #     HeadProtectNo = HeadCount[HeadBase][0]
    # else:
    #     HeadProtectNo = -1
    # if TailBase and TailNum[TailBase] == 1:
    #     TailProtectNo = TailCount[TailBase][0]
    # else:
    #     TailProtectNo = -1

    AllSecs = []
    for s, n in coefs:
        AllSecs.extend([s for _ in range(n)])
    if not (HeadBase and TailBase):
        if HeadBase:
            TailBase = HeadBase
        else:
            HeadBase = TailBase
    GoodPath = SecGraph(AllSecs, HeadBase, TailBase).search_path(REVERSE=REVERSE)
    FinalSeq = ''.join([_ if i == 0 else _[1:] for i, _ in enumerate(GoodPath)])
    return FinalSeq



