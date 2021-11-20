import numpy as np
import xarray as xr
from collections import OrderedDict as odict
import datetime
import shutil
import sys
import os
import subprocess
import socket

# Add src directory to Python path, and import DALES specific tools
src_dir = os.path.abspath('{}/../../src/'.format(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_dir)

from DALES_tools import *
from pbs_scripts import create_runscript, create_postscript


def execute_c(task):
    """
    Execute `task` and return return code
    """
    return subprocess.call(task, shell=True, executable='/bin/bash')


def execute_r(call):
    """
    Execute `task` and return output of the process (most useful here for getting the PBS job-ID)
    """
    sp = subprocess.Popen(call, shell=True, executable='/bin/bash', stdout=subprocess.PIPE)
    return sp.stdout.read().decode("utf-8").rstrip('\n')  # jikes!


def submit(script, workdir, dependency=None):
    """
    Submit a runscript (`script`) in work directory `workdir`
    If `dependency` is not None, the task is submitted but
    waits for dependency to finish
    """
    if dependency is None:
        tid = execute_r('qsub {}/{}'.format(workdir, script))
        print('Submitted {}: {}'.format(script, tid))
    else:
        tid = execute_r('qsub -W depend=afterok:{} {}/{}'.format(dependency, workdir, script))
        print('Submitted {}: {} (depends on: {})'.format(script, tid, dependency))
    return tid


def fbool(flag):
    """
    Convert a Python bool to Fortran bool
    """
    return '.true.' if flag==True else '.false.'


def calc_geo_height(ds_):
    # calculate variable z (geopotential height) from idrostatic 
    Rd = 287.06
    g = 9.8066
    rho = 100.*ds_.p/(Rd*ds_.T*(1+0.61*(ds_.q)))
    k = np.arange(ds_.level[0]+(ds_.level[1]-ds_.level[0])/2,\
                  ds_.level[-1]+(ds_.level[1]-ds_.level[0])/2,\
                  ds_.level[1]-ds_.level[0])
    rho_interp = rho.interp(level=k)
    zz = np.zeros((len(ds_['time']),len(ds_['level'])))
    zz = ((100.*ds_.p.diff(dim='level').values)/(-1*rho_interp*g)).cumsum(dim='level')
    z = zz.interp(level=ds_.level,kwargs={"fill_value": "extrapolate"})
    ds_['z']=z
    return (ds_)

def calc_q_sat(T,p):
    # calculate saturation specific humidity at the surface
    # Clausyus Clapeiron
    R  = 287.04 # J/kg*K
    Rv = 461.5 # J/kg*K
    tmelt = 273.15 # K
    bt = 35.86
    at = 17.27
    es0 = 610.78
    # Tetens Formula (as in DALES)
    es = es0 * np.exp(at*(T-tmelt) / (T-bt)) # Pa
    qs = R/Rv * es/p # kg/kg
    # approximated CC
    # es = 6.11*np.exp(0.067*(ds_.T_s-tmelt)) # hPa
    # qs = R/Rv * es*100/ds_.p_s # kg/kg
    return (qs)

if __name__ == '__main__':

    # --------------------
    # Settings
    # --------------------

    suffix     = '_300km'
    expnr       = 1       # DALES experiment number
    iloc        = 0       # Location in DDH/NetCDF files (7+12 = 10x10km average Cabauw) !!!
    dom_name    = 'BES'   # Name of the domain (it's assigned if it is not yet)
    n_accum     = 1       # Number of time steps to accumulate in the forcings
    warmstart   = False   # Run each day/run as a warm start from previous exp
    auto_submit = False   # Directly submit the experiments (ECMWF only..)

    # 24 hour runs (cold or warm starts), starting at 00 UTC.
    start  = datetime.datetime(year=2020, month=2, day=2)
    end    = datetime.datetime(year=2020, month=2, day=12)
    dt_exp = datetime.timedelta(hours=24*9)   # Time interval between experiments
    # t_exp  = datetime.timedelta(hours=24)   # Length of experiment
    t_exp  = end - start                      # Length of experiment
    eps    = datetime.timedelta(hours=1)

    # flag for including radiation to thl dynamical tendency in ls_flux
    # 0: do not include, 1: include from HARMONIE (when dtT_rad available), 2: set constant 
    harmonie_rad = 0 
    const_rad = -1/(24*3600)


    path_in = os.path.abspath('{}/../../../../../../../HARMONIE/LES_forcing_300km')
    path_out = os.path.abspath('{}/../../../../../Cases/EUREC4A')

    # ------------------------
    # End settings
    # ------------------------

    # Create stretched vertical grid for LES
    # grid = Grid_stretched(kmax=160, dz0=20, nloc1=80, nbuf1=20, dz1=150)
    grid = Grid_linear_stretched(kmax=150, dz0=20, alpha=0.012)
    # Create mixed grid (extending to ~20km with 128 gridpoints)
    # equidistant in lower 6km, linearly stretched higher up
    # grid = Grid_equidist_and_stretched(kmax=128, kloc0=75, dz0=80, alpha=0.04)    
    # grid.plot()

    
    date = start
    n = 1
    while date < end:
        print('-----------------------')
        print('Starting new experiment')
        print('-----------------------')

        # In case of warm starts, first one is still a cold one..
        start_is_warm = warmstart and n>1
        start_is_cold = not start_is_warm

        # # Round start date (first NetCDF file to read) to the 3-hourly HARMONIE cycles
        # offset = datetime.timedelta(hours=0) if date.hour%3 == 0 else datetime.timedelta(hours=-date.hour%3)
        # Round start date (first NetCDF file to read) to the 24-hourly HARMONIE cycles
        offset = datetime.timedelta(hours=0) if date.hour%24 == 0 else datetime.timedelta(hours=-date.hour%24)
        

        # Get list of NetCDF files which need to be processed, and open them with xarray
        nc_files = get_file_list(path_in, date+offset, date+t_exp+eps)
        # # manually write the files to open
        # nc_files = '{0:}/LES_forcing_{1:04d}{2:02d}{3:02d}{4:02d}.nc'.\
        #     format(path_in, date.year, date.month, date.day,date.hour)
        
        try:
            nc_data  = xr.open_mfdataset(nc_files, combine='by_coords')
        except TypeError:
            nc_data  = xr.open_mfdataset(nc_files)
            
        if 'z'      not in (list(nc_data.keys())):
                nc_data = calc_geo_height(nc_data)
        if 'domain' not in nc_data.dims:
            nc_data = nc_data.expand_dims({'domain':[dom_name]},axis=1)
        
        ##################################################################
        if suffix     == 'colder_surf':
            surf_bias = 0.5 # K
            nc_data['T_s'] -= surf_bias
        if suffix == '_p_const':
            nc_data['p_s'] = nc_data['p_s']*0 + nc_data['p_s'].mean()
        
        if 'dtql_dyn'not in (list(nc_data.keys())): # !!! 
                nc_data['dtql_dyn'] = nc_data['dtqc_dyn']
                
        if 'dtT_rad'not in (list(nc_data.keys())): # !!! WRONG
                nc_data['dtT_rad'] = nc_data['dtT_phy']*0
                
        if 'dtqv_dyn'not in (list(nc_data.keys())): # !!! 
                nc_data['dtqv_dyn'] = nc_data['dtq_dyn']
        if 'q_s'not in (list(nc_data.keys())): # !!! 
                nc_data['q_s'] = calc_q_sat(nc_data.T_s,nc_data.p_s)
        if 'th_s' not in (list(nc_data.keys())): # !!! 
                nc_data['th_s'] = nc_data['T_s']/\
                    (nc_data['p_s'] / constants['p0'])**\
                        (constants['rd']/constants['cp'])

        ##################################################################
        
        ################################################################## 
        ## The script does not need surface latent or sensible heat flux!      
        ################################################################## 
        
        
            
        # Get indices of start/end date/time in `nc_data`
        t0, t1 = get_start_end_indices(date, date + t_exp + eps, nc_data.time.values)

        # Docstring for DALES input files
        domain    = nc_data.domain.values[0]
        lat       = float(nc_data.central_lat)
        lon       = float(nc_data.central_lon)
        docstring = '{0} ({1:.2f}N, {2:.2f}E): {3} to {4}'.format(domain, lat, lon, date, date + t_exp)

        if start_is_cold:
            # Create and write the initial vertical profiles (prof.inp)
            create_initial_profiles(nc_data, grid, t0, t1, iloc, docstring, expnr)

        # Create and write the surface and atmospheric forcings (ls_flux.inp, ls_fluxsv.inp, lscale.inp)
        # lscale.inp is filled with zero just because DALES needs it.
        create_ls_forcings(nc_data, grid, t0, t1, iloc, docstring, n_accum, expnr,\
                           harmonie_rad=harmonie_rad,const_rad=const_rad)

        # Write the nudging profiles (nudge.inp) 
        nudgefac = np.ones_like(grid.z)     # ?? -> set to zero in ABL?
        create_nudging_profiles(nc_data, grid, nudgefac, t0, t1, iloc, docstring, 1, expnr)

        # Create NetCDF file with reference/background profiles for RRTMG
        # In future make backrad time dependent 
        t_back = t0 # for now, arbitrary choose a time from harmonie 
        create_backrad(nc_data, t_back, iloc, expnr)


        # Update namelist
        namelist = 'namoptions.{0:03d}'.format(expnr)
        replace_namelist_value(namelist, 'lwarmstart', fbool(start_is_warm))
        replace_namelist_value(namelist, 'iexpnr',   '{0:03d}'.format(expnr))
        replace_namelist_value(namelist, 'runtime',  t_exp.total_seconds())
        replace_namelist_value(namelist, 'trestart', t_exp.total_seconds())
        replace_namelist_value(namelist, 'xlat',     lat)
        replace_namelist_value(namelist, 'xlon',     lon)
        replace_namelist_value(namelist, 'xday',     date.timetuple().tm_yday)
        replace_namelist_value(namelist, 'xtime',    date.hour)
        replace_namelist_value(namelist, 'kmax',     grid.kmax)

        # Read back namelist
        nl = Read_namelist('namoptions.{0:03d}'.format(expnr))

        # Copy/move files to work directory
        workdir = '{0}/{1:04d}{2:02d}{3:02d}_{4:02d}'.format(path_out, date.year, date.month, date.day,(date+t_exp).day)
        workdir += suffix
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        # Create SLURM runscript
        # print('Creating runscript')
        # ntasks = nl['run']['nprocx']*nl['run']['nprocy']
        # create_runscript ('L{0:03d}_{1}'.format(expnr, n), ntasks, walltime=24, work_dir=workdir, expnr=expnr)

        # Copy/move files to work directory
        exp_str = '{0:03d}'.format(expnr)
        to_copy = ['rrtmg_lw.nc', 'rrtmg_sw.nc', #'dales4','namoptions.{}'.format(exp_str),
                   'prof.inp.{}'.format(exp_str), 'scalar.inp.{}'.format(exp_str), 'mergecross.py']
        to_move = ['backrad.inp.{}.nc'.format(exp_str), 'lscale.inp.{}'.format(exp_str),\
                   'ls_flux.inp.{}'.format(exp_str), 'ls_fluxsv.inp.{}'.format(exp_str),\
                   'nudge.inp.{}'.format(exp_str)] #, 'run.PBS']

        print('Copying/moving input files')
        for f in to_move:
            shutil.move(f, '{}/{}'.format(workdir, f))
        for f in to_copy:
            shutil.copy(f, '{}/{}'.format(workdir, f))

        if start_is_warm:
            # Link base state and restart files from `prev_wdir` to the current working directory)
            print('Creating symlinks to restart files')

            hh = int(t_exp.total_seconds()/3600)
            mm = int(t_exp.total_seconds()-(hh*3600))

            # Link base state profile
            f_in  = '{0}/baseprof.inp.{1:03d}'.format(prev_workdir, expnr)
            f_out = '{0}/baseprof.inp.{1:03d}'.format(workdir, expnr)

            if not os.path.exists(f_out):
                os.symlink(f_in, f_out)

            # Link restart files
            for i in range(nl['run']['nprocx']):
                for j in range(nl['run']['nprocy']):
                    for ftype in ['d','s','l']:

                        f_in  = '{0}/init{1}{2:03d}h{3:02d}mx{4:03d}y{5:03d}.{6:03d}'\
                                    .format(prev_workdir, ftype, hh, mm, i, j, expnr)
                        f_out = '{0}/init{1}000h00mx{2:03d}y{3:03d}.{4:03d}'\
                                    .format(workdir, ftype, i, j, expnr)

                        if not os.path.exists(f_out):
                            os.symlink(f_in, f_out)

        # Submit task, accounting for job dependencies in case of warm start
        if auto_submit:
            if start_is_warm:
                run_id = submit('run.PBS', workdir, dependency=prev_run_id)
            else:
                run_id = submit('run.PBS', workdir)

        # Create and submit post-processing task
        create_postscript('P{0:03d}_{1}'.format(expnr, n), walltime=24, work_dir=workdir, expnr=expnr,
                itot=nl['domain']['itot'], jtot=nl['domain']['jtot'], ktot=nl['domain']['kmax'], 
                nprocx=nl['run']['nprocx'], nprocy=nl['run']['nprocy'])

        shutil.move('post.PBS', '{}/post.PBS'.format(workdir))

        if auto_submit:
            post_id = submit('post.PBS', workdir, dependency=run_id)

        # Advance time and store some settings
        date += dt_exp
        n += 1
        prev_workdir = workdir

        if auto_submit:
            prev_run_id = run_id
