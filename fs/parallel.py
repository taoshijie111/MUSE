#!/usr/bin/env python
'''
Author: Ifsoul
Date: 2020-08-29 16:11:47
LastEditTime: 2021-04-12 15:20:48
LastEditors: Ifsoul
Description: Functions for parallel running
'''
import os
# import sys
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import subprocess as sb
import numpy as np
import time
import psutil
import signal

from .core import find_same, sec2time
from .plot import fig_double_y


class job_info(object):
    """Get environment variables of current job"""

    def __init__(self):
        if "SLURM_JOBID" in os.environ:
            self.Type = 'SLURM'
            self.JobId = int(os.environ["SLURM_JOBID"])
            self.PId = int(os.environ["SLURM_TASK_PID"])
            self.Queue = os.environ["SLURM_JOB_PARTITION"]
            self.Name = os.environ["SLURM_JOB_NAME"]
            if "SLURM_NTASKS" in os.environ:
                self.MaxCPU = int(os.environ["SLURM_NTASKS"])
            elif "SLURM_NTASKS_PER_NODE" in os.environ:
                self.MaxCPU = int(os.environ["SLURM_NTASKS_PER_NODE"])
            elif "SLURM_CPUS_PER_TASK" in os.environ:
                self.MaxCPU = int(os.environ["SLURM_CPUS_PER_TASK"])
            else:
                self.MaxCPU = 4
        elif "LSB_JOBID" in os.environ:
            self.Type = 'LSF'
            self.JobId = int(os.environ["LSB_JOBID"])
            self.PId = int(os.environ["LS_JOBPID"])
            self.Queue = os.environ["LSB_QUEUE"]
            self.Name = os.environ["LSB_JOBNAME"]
            self.MaxCPU = int(os.environ["LSB_MAX_NUM_PROCESSORS"])
        # elif "PBS_JOBID" in os.environ:
        #     self.Type='PBS'
        #     self.JobId=os.environ["PBS_JOBID"]
        else:
            self.Type = 'UNKOWN'
            self.JobId = os.getpid()
            self.PId = os.getpid()
            self.Queue = 'UNKOWN'
            self.Name = 'UNKOWN'
            self.MaxCPU = int(os.environ["NUMBER_OF_PROCESSORS"]) if "NUMBER_OF_PROCESSORS" in os.environ else mp.cpu_count()
        self.CPUAvail = self.MaxCPU
        self.TotalMem = psutil.virtual_memory().total

    def info(self):
        """Print infos"""
        print(
            'Job Info:\n',
            'Type:    %s\n' % self.Type,
            'Name:    %s\n' % self.Name,
            'Queue:   %s\n' % self.Queue,
            'JobId:   %d\n' % self.JobId,
            'PId:     %d\n' % self.PId,
            'MaxCPU:  %d\n' % self.MaxCPU,
            'MaxMeM:  %-.2f GB\n' % (self.TotalMem / 1073741824),
        )


Job = job_info()
# Job.info()


def get_info_func(pid=None):
    if pid is None:

        def f():
            return psutil.cpu_percent(), psutil.virtual_memory().used / 1073741824

        return f
    else:

        def f():
            MyProc = psutil.Process(pid)
            MyProc.cpu_percent()
            mem = MyProc.memory_info().rss
            CProcs = MyProc.children(recursive=True)
            for p in CProcs:
                # if p.status()==psutil.STATUS_RUNNING:
                try:
                    p.cpu_percent()
                    mem += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            time.sleep(0.2)
            cpu = MyProc.cpu_percent()
            for p in CProcs:
                try:
                    cpu += p.cpu_percent()
                except psutil.NoSuchProcess:
                    pass
            # return psutil.Process(pid).memory_info().vms / 1073741824
            return cpu, mem / 1073741824

        return f


class monitor(object):
    """Monitor of the CPU and memory used by current job"""

    def __init__(self, name='', title='', sleeptime=1, SELFINFO=True):
        self.Name = name
        self.title = title
        self.SleepTime = max(0.1, sleeptime - 0.2)
        self.pid = os.getpid()
        self.Node = []
        self.InfoFunc = get_info_func(self.pid) if SELFINFO else get_info_func()

    def __enter__(self):
        self.Start = time.time()
        self.Event = mp.Event()
        self.Mgr = mp.Manager()
        self.Info = self.Mgr.list()
        self.Worker = mp.Process(target=self.run, args=(
            self.Event,
            self.Info,
            self.Start,
            self.InfoFunc,
            self.SleepTime,
        ), daemon=True)
        self.Worker.start()
        return self

    def __exit__(self, *args):
        self.Event.set()
        self.Worker.join()
        self.End = time.time() - self.Start
        print('%s time: %s' % (self.Name, sec2time(self.End)))
        self.plot()

    @staticmethod
    def run(StopEvent, Info, StartTime, InfoFunc, SleepTime=10):
        """Run monitor"""
        while True:
            Info.append((time.time() - StartTime, *InfoFunc()))
            if StopEvent.is_set():
                return
            else:
                time.sleep(SleepTime)

    def add_node(self, name=''):
        """Add node"""
        self.Node.append((time.time() - self.Start, name))

    def plot(self):
        """Plot Info as a figure"""
        FigName = self.Name + '.png'
        Title = self.title
        Labels = ('CPU', 'Memory')
        AxisTitle = ('Time (s)', 'CPU used (%)', 'Memory used (GB)')
        Styles = [{'c': 'b'}, {'c': 'r'}]
        fig_double_y(
            FigName,
            np.array(self.Info),
            Labels=Labels,
            Title=Title,
            AxisTitle=AxisTitle,
            XLimit=(0, self.End),
            YLine=self.Node,
            Styles=Styles,
        )


def signal_handler(signum, frame):
    # handle exit
    signame = signal.Signals(signum).name
    print(f'Signal handler called with signal {signame} ({signum})')
    if os.name == 'nt':
        os.kill(os.getpgid(os.getpid()), signal.CTRL_C_EVENT)
    else:
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def run_functions(funcs, PoolSize=Job.MaxCPU, SubPoolSize=1, PoolKwargs={}, ApplyKwargs={}):
    """Run functions parallelly"""
    ParentCPUs, Job.CPUAvail = Job.CPUAvail, SubPoolSize
    assert PoolSize > 0, "PoolSize should be larger than 0!"
    NPool = min(PoolSize, len(funcs))
    if NPool > 0:
        pool = mp.Pool(NPool, **PoolKwargs)
        res = [pool.apply_async(*func_and_args, **ApplyKwargs) for func_and_args in funcs]
        pool.close()
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        pool.join()
        Job.CPUAvail = ParentCPUs
        return [x.get() for x in res]
    else:
        print('WARNING: Function list is empty! No function to run.')
        return []


def run_functions_thread(funcs, PoolSize=Job.MaxCPU, SubPoolSize=1, PoolKwargs={}, ApplyKwargs={}):
    """Run functions parallelly"""
    ParentCPUs, Job.CPUAvail = Job.CPUAvail, SubPoolSize
    assert PoolSize > 0, "PoolSize should be larger than 0!"
    NPool = min(PoolSize, len(funcs))
    if NPool > 0:
        pool = ThreadPool(NPool, **PoolKwargs)
        res = [pool.apply_async(*func_and_args, **ApplyKwargs) for func_and_args in funcs]
        pool.close()
        pool.join()
        Job.CPUAvail = ParentCPUs
        return [x.get() for x in res]
    else:
        print('WARNING: Function list is empty! No function to run.')
        return []


def check_process(p):
    """Wait for process to finish and check the status"""
    try:
        stdout, stderr = p.communicate()
    except BaseException:
        p.kill()
        raise
    return stdout, stderr


def run_commands(cmds, PoolSize=Job.MaxCPU, SetOMPTreads=None):
    """Run commands parallelly"""
    assert PoolSize > 0, "PoolSize should be larger than 0!"
    Ncmds = len(cmds)
    if os.name == 'nt':
        SHELL = False
        cmds = ['powershell ' + c for c in cmds]
    else:
        SHELL = True
    if SetOMPTreads is not None and "OMP_NUM_THREADS" in os.environ:
        OMPSetting, os.environ["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"], "%d" % SetOMPTreads
    Proc = []
    try:
        if Ncmds <= PoolSize:
            for c in cmds:
                Proc.append(sb.Popen(c, shell=SHELL))
            for i, p in enumerate(Proc):
                stdout, stderr = check_process(p)
                retcode = p.poll()
                if retcode or stderr:
                    print('Fail at line %d:' % (i))
                    print('%s' % (cmds[i]))
                    raise sb.CalledProcessError(retcode, p.args)
        else:
            ProcNo = [_ for _ in range(PoolSize)]
            for c in cmds[:PoolSize]:
                Proc.append(sb.Popen(c, shell=SHELL))
            NNext = PoolSize
            NDone = 0
            while NDone < Ncmds:
                while len(Proc) < PoolSize and NNext < Ncmds:
                    Proc.append(sb.Popen(cmds[NNext], shell=SHELL))
                    ProcNo.append(NNext)
                    NNext += 1
                while True:
                    NWait = len(Proc)
                    if NWait == 0:
                        break
                    stat = []
                    for k in range(NWait):
                        stat.append(Proc[k].poll())
                    if (stat.count(None) == NWait):
                        time.sleep(0.02)
                    else:
                        for k in range(NWait - 1, -1, -1):
                            if stat[k] == 0:
                                p = Proc[k]
                                stdout, stderr = check_process(p)
                                if stderr:
                                    print('Fail at line %d:' % (k))
                                    print('%s' % (cmds[k]))
                                    raise sb.CalledProcessError(stat[k], p.args)
                                NDone += 1
                                # print('Line %d finished.' % (ProcNo[k]))
                                del Proc[k]
                                del ProcNo[k]
                            elif stat[k] is not None:
                                print('Fail at line %d:' % (ProcNo[k]))
                                print('%s' % (cmds[ProcNo[k]]))
                                raise sb.CalledProcessError(stat[k], Proc[k].args)
                        break
    except BaseException:
        raise
    finally:
        if SetOMPTreads is not None and "OMP_NUM_THREADS" in os.environ:
            os.environ["OMP_NUM_THREADS"] = OMPSetting


def parallel_search(List_Like, PoolSize=Job.CPUAvail):
    """Run find_same function parallelly for all elements in List_Like, return idx_same list"""
    idx_same = run_functions([(find_same, (List_Like, i)) for i in range(len(List_Like))], PoolSize)
    return idx_same
