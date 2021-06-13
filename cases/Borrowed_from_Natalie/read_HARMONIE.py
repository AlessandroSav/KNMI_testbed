import xarray as xr 
import os 
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import netCDF4 as nc4

"""
List of HARMONIE variables needed in for LES forcings

State variables 
ta   - Absolute temperature [K]
hus  - Specific humidity [kg kg-1]
ua   - u (east-west) componant of the wind [m s-1]  
va   - v (north-south) componant of the wind [m s-1]  
cli  - liquid cloud ice mass fraction[kg kg-1] 
clw  - liquid cloud water mass fraction [kg kg-1] 

Forcings
note- forcings only saved for small Netherlands domain in monthly files and are processed in a different
      routine from the other variables 
dtT - tendencies for temperature [K s-1]
      consisting of dtT_dyn -dynamic and dtT_phys - physical tendencies
dtq - tendencies for humidity [kg kg-1 s-1]
      consisting of dtq_dyn -dynamic and dtq_phys - physical tendencies
dtqc- tendencies for cloud liquid water [kg kg-1 s-1]
      consisting of dtq_dyn -dynamic and dtq_phys - physical tendencies
dtu - tendencies for u (east-west)-componant of the wind
      consisting of dtu_dyn -dynamic and dtu_phys - physical tendencies
dtv - tendencies for v (north-south)-componant of the wind
      consisting of dtv_dyn -dynamic and dtv_phys - physical tendencies

(Near) surface variables 
ps  - surface pressure 
hfss - accumulated surface sensible heat flux 
hfls_sbl - accumulated latent heat flux through sublimation
hfls_eva - accumulated latent heat flux through evaporation
TODO!! Add Ts & qs 


fp 3d variables 
phi - geopotential height


"""
# Ignore this is just to pull data off ecmwf tapes
def get_from_ecfs_daily(date):
     """

     Get HARMONIE daily data from ECFS
     Ony works when running on ECMWF machine
     date - should be a datetime array

     """

     print('Getting daily HAROMONIE files from ECFS for {}'.format(date))
     exp = "ruisdael_IOP2020"

     vars_nearsurf = ['hfss','hfls_eva','hfls_sbl','ps','ts']
     vars_3d_state = ['hus', 'cli', 'clw', 'ta','ua','va']
     vars_fp = ['phi']

     path = 'ec:/nknt/harmonie/ruisdael/{0}/{1:04d}/{2:02d}/{3:02d}/00/'.\
                format(exp,date.year,date.month,date.day)
     curr_dir = os.path.dirname(os.path.realpath(__file__))
     print(curr_dir)
     for varstr in vars_3d_state:

         f = '{0}.Slev.his.NETHERLANDS.{1}.{2:04d}{3:02d}{4:02d}.1hr.nc'.\
                   format(varstr,exp, date.year, date.month, date.day)
         print(curr_dir+'/'+f)
         if os.path.exists(curr_dir+'/'+f):
             print('Already have file: {0}'.format(f))
         else:
             print('getting file: {0}'.format(f))
             os.system('ecp {0}{1} . '.format(path,f))

     for varstr in vars_nearsurf:

         f = '{0}.his.NETHERLANDS.{1}.{2:04d}{3:02d}{4:02d}.1hr.nc'.\
                      format(varstr,exp, date.year, date.month, date.day)
    
         if os.path.exists(curr_dir+'/'+f):
             print('Already have file: {0}'.format(f))
         else:
             print('getting file: {0}'.format(f))
             os.system('ecp {0}{1} . '.format(path,f))

     for varstr in vars_fp:

         f = '{0}.Slev.fp.NETHERLANDS.{1}.{2:04d}{3:02d}{4:02d}.1hr.nc'.\
                      format(varstr,exp, date.year, date.month, date.day)
    
         if os.path.exists(curr_dir+'/'+f):
             print('Already have file: {0}'.format(f))
         else:
             print('getting file: {0}'.format(f))
             os.system('ecp {0}{1} . '.format(path,f))

def get_from_ecfs_monthly(date):
     """
     Get HARMONIE monthly data from ECFS 
     Ony works when running on ECMWF machine
     date - should be a datetime array 
  
     """
     print('Getting montly HARMONIE files from ECFS for {}-{}'.format(date.year,date.month))
 
     exp = "ruisdael_IOP2020"
     vars_tend = ['dtq_dyn', 'dtq_phy', 'dtT_dyn', 'dtT_phy', 'dtu_dyn', 'dtu_phy', 'dtv_dyn', 'dtv_phy', 'dtqc_dyn', 'dtqc_phy']
     path = 'ec:/nknt/harmonie/ruisdael/{0}/{1:04d}/{2:02d}/01/00/'.\
                format(exp,date.year,date.month)
     curr_dir = os.path.dirname(os.path.realpath(__file__))

     for varstr in vars_tend:

         f = '{0}.Slev.his.NETHERLANDS.NL.{1}.{2:04d}{3:02d}.1hr.nc'.\
                   format(varstr,exp, date.year, date.month)

         if os.path.exists(curr_dir+'/'+f):
             print('Already have file: {0}'.format(f))
         else:
             print('getting file: {0}'.format(f))
             os.system('ecp {0}{1} . '.format(path,f))


class harmonie_to_LESforce:

    def __init__(self, path, exp, date, lat, lon, buffer=0, add_soil=False, dt=60, quiet=False):
        ahalf= (pd.read_csv('H43lev65.txt',header=None,index_col=[0],delim_whitespace=True))[1].values[:]
        bhalf= (pd.read_csv('H43lev65.txt',header=None,index_col=[0],delim_whitespace=True))[2].values[:]
        
        self.nlev  = len(bhalf) - 1     # Number of full vertical levels
        self.nlevh = self.nlev + 1      # Number of half vertical levels
        self.nt    = int(25)            # Number of output time steps

        # Create empty arrays to store the individual DDH data
        self.time = np.ma.zeros(self.nt)
        self.hour = np.ma.zeros(self.nt)
        self.datetime = []
        self.lat = lat
        self.lon = lon

        # Array dimensions
        dim3d  = (self.nt, self.nlev)
        dim3dh = (self.nt, self.nlevh)
        dim2d  = (self.nt)


        # Atmospheric quantities
        # ----------------------
        self.cp   = np.ma.zeros(dim3d)  # Specific heat at const pressure (J kg-1 K-1)
        self.p    = np.ma.zeros(dim3d)  # Pressure (Pa)
        self.dp   = np.ma.zeros(dim3d)  # Pressure difference (Pa)
        self.z    = np.ma.zeros(dim3d)  # Geopotential height (m)
        self.ph   = np.ma.zeros(dim3dh) # Half level pressure (Pa)
        self.zh   = np.ma.zeros(dim3dh) # Half level geopotential height (m)

        self.u    = np.ma.zeros(dim3d)  # u-component wind (m s-1)
        self.v    = np.ma.zeros(dim3d)  # v-component wind (m s-1)
        self.T    = np.ma.zeros(dim3d)  # Absolute temperature (K)
        self.q    = np.ma.zeros(dim3d)  # Specific humidity (kg kg-1)
        self.qcw    = np.ma.zeros(dim3d)  # liquid cloud water mass fraction[kg kg-1] 
        self.qci    = np.ma.zeros(dim3d)  # cloud ice mass fraction[kg kg-1] 

        # Surface quantities
        # ----------------------
        self.H    = np.ma.zeros(dim2d)  # Surface sensible heat flux (W m-2)
        self.LE   = np.ma.zeros(dim2d)  # Surface latent heat flux (W m-2)
        self.Tsk  = np.ma.zeros(dim2d)  # Surface temperature (K)
        self.qsk  = np.ma.zeros(dim2d)  # Surface specific humidity (kg kg-1)
        self.ps   = np.ma.zeros(dim2d)  # Surface Pressure (Pa)
        self.swds = np.ma.zeros(dim2d)  # Surface incoming shortwave radiation (W m-2)
        self.lwds = np.ma.zeros(dim2d)  # Surface incoming longwave radiation (W m-2)

        # Physics, dynamics and total tendencies
        # Units all in "... s-1"
        self.dtu_phy = np.ma.zeros(dim3d)
        self.dtv_phy = np.ma.zeros(dim3d)
        self.dtT_phy = np.ma.zeros(dim3d)
        self.dtq_phy = np.ma.zeros(dim3d)
        self.dtqc_phy = np.ma.zeros(dim3d)

        self.dtu_dyn = np.ma.zeros(dim3d)
        self.dtv_dyn = np.ma.zeros(dim3d)
        self.dtT_dyn = np.ma.zeros(dim3d)
        self.dtq_dyn = np.ma.zeros(dim3d)
        self.dtqc_dyn = np.ma.zeros(dim3d)

        self.dtu_tot = np.ma.zeros(dim3d)
        self.dtv_tot = np.ma.zeros(dim3d)
        self.dtT_tot = np.ma.zeros(dim3d)
        self.dtq_tot = np.ma.zeros(dim3d)
        self.dtqc_tot = np.ma.zeros(dim3d)

        """
        Read out tendencies from HARMONIE netcdf files
        """

        # Read in the tendencies first 
        if True: 
            tend_names = ['dtq_dyn', 'dtq_phy', 'dtT_dyn', 'dtT_phy', 'dtu_dyn', 'dtu_phy', 'dtv_dyn', 'dtv_phy', 'dtqc_dyn', 'dtqc_phy']

            files = []
            for varstr in tend_names:

                f = '{0}{1}.Slev.his.NETHERLANDS.NL.{2}.{3:04d}{4:02d}.1hr.nc'.\
                format(path,varstr, exp, date.year, date.month)

                if os.path.exists(f):
                    files.append(f)
                else:
                    print('Can not find {}!! Skipping..'.format(f))

            nc = xr.open_mfdataset(files)
            nc = nc.sel(time = date.strftime("%Y-%m-%d"))
            hm_lat = nc.lat.values
            hm_lon = nc.lon.values

            j,i = np.unravel_index(np.sqrt((hm_lon-lon)**2 + (hm_lat-lat)**2).argmin(), hm_lon.shape)


            if date.day==1:  #the first timestep in every (month-) file is useless. 
                nc = nc.isel(time=slice(1,None))
            if (buffer>0) : 
                dtq_dynacc = np.nanmean(nc.dtq_dyn.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                dtq_phyacc = np.nanmean(nc.dtq_phy.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                dtT_dynacc = np.nanmean(nc.dtT_dyn.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                dtT_phyacc = np.nanmean(nc.dtT_phy.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                dtu_dynacc = np.nanmean(nc.dtu_dyn.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                dtu_phyacc = np.nanmean(nc.dtu_phy.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                dtv_dynacc = np.nanmean(nc.dtv_dyn.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                dtv_phyacc = np.nanmean(nc.dtv_phy.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                dtqc_dynacc = np.nanmean(nc.dtqc_dyn.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                dtqc_phyacc = np.nanmean(nc.dtqc_phy.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
            else : 
                dtq_dynacc = nc.dtq_dyn.values[:,:,j,i]
                dtq_phyacc = nc.dtq_phy.values[:,:,j,i]
                dtT_dynacc = nc.dtT_dyn.values[:,:,j,i]
                dtT_phyacc = nc.dtT_phy.values[:,:,j,i]
                dtu_dynacc = nc.dtu_dyn.values[:,:,j,i]
                dtu_phyacc = nc.dtu_phy.values[:,:,j,i]
                dtv_dynacc = nc.dtv_dyn.values[:,:,j,i]
                dtv_phyacc = nc.dtv_phy.values[:,:,j,i]
                dtqc_dynacc = nc.dtqc_dyn.values[:,:,j,i]
                dtqc_phyacc = nc.dtqc_phy.values[:,:,j,i]

            # deaccumulate for timebounds differences of more than 1 hour 
            for t in range(0,self.nt-1):

                time_bnds = nc.time_bnds.values[t,:]
                timediff = (time_bnds[-1]-time_bnds[0]).astype('timedelta64[s]') # convert to seconds

                if timediff <= timedelta(seconds=3600):
                    self.dtq_dyn[t+1,:] = dtq_dynacc[t,:]/dt
                    self.dtq_phy[t+1,:] = dtq_phyacc[t,:]/dt
                    self.dtT_dyn[t+1,:] = dtT_dynacc[t,:]/dt
                    self.dtT_phy[t+1,:] = dtT_phyacc[t,:]/dt
                    self.dtu_dyn[t+1,:] = dtu_dynacc[t,:]/dt
                    self.dtu_phy[t+1,:] = dtu_phyacc[t,:]/dt
                    self.dtv_dyn[t+1,:] = dtv_dynacc[t,:]/dt
                    self.dtv_phy[t+1,:] = dtv_phyacc[t,:]/dt
                    self.dtqc_dyn[t+1,:] = dtqc_dynacc[t,:]/dt
                    self.dtqc_phy[t+1,:] = dtqc_phyacc[t,:]/dt
                else: 
                    self.dtq_dyn[t+1,:] = (dtq_dynacc[t,:] - dtq_dynacc[t-1,:])/dt
                    self.dtq_phy[t+1,:] = (dtq_phyacc[t,:] - dtq_phyacc[t-1,:])/dt
                    self.dtT_dyn[t+1,:] = (dtT_dynacc[t,:] - dtT_dynacc[t-1,:])/dt
                    self.dtT_phy[t+1,:] = (dtT_phyacc[t,:] - dtT_phyacc[t-1,:])/dt
                    self.dtu_dyn[t+1,:] = (dtu_dynacc[t,:] - dtu_dynacc[t-1,:])/dt
                    self.dtu_phy[t+1,:] = (dtu_phyacc[t,:] - dtu_phyacc[t-1,:])/dt
                    self.dtv_dyn[t+1,:] = (dtv_dynacc[t,:] - dtv_dynacc[t-1,:])/dt
                    self.dtv_phy[t+1,:] = (dtv_phyacc[t,:] - dtv_phyacc[t-1,:])/dt     
                    self.dtqc_dyn[t+1,:] = (dtqc_dynacc[t,:] - dtqc_dynacc[t-1,:])/dt
                    self.dtqc_phy[t+1,:] = (dtqc_phyacc[t,:] - dtqc_phyacc[t-1,:])/dt

            time_bnds = nc.time_bnds.values

            # Generate list of NetCDF files to read
            # other 3D fields 
            vars_3d_state = ['hus', 'cli', 'clw', 'ta','ua','va']
            files = []
            for varstr in vars_3d_state:

                f = '{0}{1}.Slev.his.NETHERLANDS.{2}.{3:04d}{4:02d}{5:02d}.1hr.nc'.\
                format(path,varstr,exp, date.year, date.month, date.day)

                if os.path.exists(f):
                    files.append(f)
                else:
                    print('Can not find {}!! Skipping..'.format(f))

            nc = xr.open_mfdataset(files)
            #nc = nc.sel(time = date.strftime("%Y-%m-%d"))
            hm_lat = nc.lat.values
            hm_lon = nc.lon.values    

            j,i = np.unravel_index(np.sqrt((hm_lon-lon)**2 + (hm_lat-lat)**2).argmin(), hm_lon.shape)
            
            print(j-buffer,j+buffer,i-buffer,i+buffer)
            if (buffer>0) : 
                self.u = np.nanmean(nc.ua.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                self.v = np.nanmean(nc.va.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                self.T = np.nanmean(nc.ta.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                self.q = np.nanmean(nc.hus.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                self.qcw = np.nanmean(nc.clw.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
                self.qci = np.nanmean(nc.cli.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))

            else : 
                self.u = nc.ua.values[:,:,j,i]
                self.v = nc.va.values[:,:,j,i]
                self.T = nc.ta.values[:,:,j,i]
                self.q = nc.hus.values[:,:,j,i]
                self.qcw = nc.clw.values[:,:,j,i]
                self.qci = nc.cli.values[:,:,j,i]

            self.time = nc.time.values
            self.hour = nc.time.dt.hour
            self.time    = np.array(self.time)
            self.hours_since = np.array([(t-np.datetime64('2010-01-01T00:00:00')).astype('timedelta64[s]')/3600. for t in self.time])
            vars_fp3d = ['phi']
            # Generate list of NetCDF files to read
            # 3d fp fields
            files = []
            for varstr in vars_fp3d:

                f = '{0}{1}.Slev.fp.NETHERLANDS.{2}.{3:04d}{4:02d}{5:02d}.1hr.nc'.\
                format(path,varstr,exp, date.year, date.month, date.day)

                if os.path.exists(f):
                    files.append(f)
                else:
                    print('Can not find {}!! Skipping..'.format(f))

            nc = xr.open_mfdataset(files)
            #nc = nc.sel(time = date.strftime("%Y-%m-%d"))
            hm_lat = nc.lat.values
            hm_lon = nc.lon.values    

            j,i = np.unravel_index(np.sqrt((hm_lon-lon)**2 + (hm_lat-lat)**2).argmin(), hm_lon.shape)

            if (buffer>0) : 
                self.z = np.nanmean(nc.phi.values[:,:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(2,3))
            else : 
                self.z = nc.phi.values[:,:,j,i]


            # Generate list of NetCDF files to read
            # 2d fields (surface - accumulated)
            vars_nearsurf = ['hfss','hfls_eva','hfls_sbl']

            files = []
            for varstr in vars_nearsurf:

                f = '{0}{1}.his.NETHERLANDS.{2}.{3:04d}{4:02d}{5:02d}.1hr.nc'.\
                format(path,varstr,exp, date.year, date.month, date.day)

                if os.path.exists(f):
                    files.append(f)
                else:
                    print('Can not find {}!! Skipping..'.format(f))

            nc = xr.open_mfdataset(files)
           # nc = nc.sel(time = date.strftime("%Y-%m-%d"))
            hm_lat = nc.lat.values
            hm_lon = nc.lon.values    

            j,i = np.unravel_index(np.sqrt((hm_lon-lon)**2 + (hm_lat-lat)**2).argmin(), hm_lon.shape)

            if (buffer>0) : 
                hfss = np.nanmean(nc.hfss.values[:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(1,2))
                hfls = (np.nanmean(nc.hfls_eva.values[:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(1,2))+
                           np.nanmean(nc.hfls_sbl.values[:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(1,2)))

            else : 
                hfss = nc.hfss.values[:,j,i]
                hfls = (nc.hfls_eva.values[:,j,i] + nc.hfls_sbl.values[:,j,i])

            # Generate list of NetCDF files to read
            # 2d fields (surface - NOT accumulated)
            vars_nearsurf = ['ps', 'ts']

            files = []
            for varstr in vars_nearsurf:

                f = '{0}{1}.his.NETHERLANDS.{2}.{3:04d}{4:02d}{5:02d}.1hr.nc'.\
                format(path,varstr,exp, date.year, date.month, date.day)

                if os.path.exists(f):
                    files.append(f)
                else:
                    print('Can not find {}!! Skipping..'.format(f))

            nc = xr.open_mfdataset(files)
            #nc = nc.sel(time = date.strftime("%Y-%m-%d"))
            hm_lat = nc.lat.values
            hm_lon = nc.lon.values

            j,i = np.unravel_index(np.sqrt((hm_lon-lon)**2 + (hm_lat-lat)**2).argmin(), hm_lon.shape)

            if (buffer>0) :
                self.ps = np.nanmean(nc.ps.values[:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(1,2))
                self.Tsk = np.nanmean(nc.ts.values[:,j-buffer:j+buffer,i-buffer:i+buffer],axis=(1,2))

            else :
                self.ps = nc.ps.values[:,j,i]
                self.Tsk = nc.ts.values[:,j,i]

            ph = np.array([ahalf + (p * bhalf) for p in self.ps ])
            for z in range(0,self.nlev):
                self.p[:,z] = 0.5 * (ph[:,z] + ph[:,z+1])

            # deaccumulate for timebounds differences of more than 1 hour 
            for t in range(0,self.nt-1):

                time_bnd = time_bnds[t,:]
                timediff = (time_bnd[-1]-time_bnd[0]).astype('timedelta64[s]') # convert to seconds

                if timediff <= timedelta(seconds=3600):
                    self.dtq_dyn[t+1,:] = dtq_dynacc[t,:]/dt
                    self.dtq_phy[t+1,:] = dtq_phyacc[t,:]/dt
                    self.dtT_dyn[t+1,:] = dtT_dynacc[t,:]/dt
                    self.dtT_phy[t+1,:] = dtT_phyacc[t,:]/dt
                    self.dtu_dyn[t+1,:] = dtu_dynacc[t,:]/dt
                    self.dtu_phy[t+1,:] = dtu_phyacc[t,:]/dt
                    self.dtv_dyn[t+1,:] = dtv_dynacc[t,:]/dt
                    self.dtv_phy[t+1,:] = dtv_phyacc[t,:]/dt
                    self.dtqc_dyn[t+1,:] = dtqc_dynacc[t,:]/dt
                    self.dtqc_phy[t+1,:] = dtqc_phyacc[t,:]/dt
                else:
                    self.dtq_dyn[t+1,:] = (dtq_dynacc[t,:] - dtq_dynacc[t-1,:])/dt
                    self.dtq_phy[t+1,:] = (dtq_phyacc[t,:] - dtq_phyacc[t-1,:])/dt
                    self.dtT_dyn[t+1,:] = (dtT_dynacc[t,:] - dtT_dynacc[t-1,:])/dt
                    self.dtT_phy[t+1,:] = (dtT_phyacc[t,:] - dtT_phyacc[t-1,:])/dt
                    self.dtu_dyn[t+1,:] = (dtu_dynacc[t,:] - dtu_dynacc[t-1,:])/dt
                    self.dtu_phy[t+1,:] = (dtu_phyacc[t,:] - dtu_phyacc[t-1,:])/dt
                    self.dtv_dyn[t+1,:] = (dtv_dynacc[t,:] - dtv_dynacc[t-1,:])/dt
                    self.dtv_phy[t+1,:] = (dtv_phyacc[t,:] - dtv_phyacc[t-1,:])/dt
                    self.dtqc_dyn[t+1,:] = (dtqc_dynacc[t,:] - dtqc_dynacc[t-1,:])/dt
                    self.dtqc_phy[t+1,:] = (dtqc_phyacc[t,:] - dtqc_phyacc[t-1,:])/dt
                print(hfss.shape)
                self.H[t+1] = hfss[t]/dt
                self.LE[t+1] = hfls[t]/dt

        # Check...: sum of dyn+phys
        self.dtu_sum = self.dtu_phy + self.dtu_dyn
        self.dtv_sum = self.dtv_phy + self.dtv_dyn
        self.dtT_sum = self.dtT_phy + self.dtT_dyn
        self.dtq_sum = self.dtq_phy + self.dtq_dyn

        # Check...: offline tendency
        self.dtu_off  = self.calc_tendency(self.u,  dt)
        self.dtv_off  = self.calc_tendency(self.v,  dt)
        self.dtT_off  = self.calc_tendency(self.T,  dt)
        self.dtq_off  = self.calc_tendency(self.q , dt)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(self.time, self.dtT_sum[:,50])
        # plt.plot(self.time, self.dtT_off[:,50])
        # plt.figure()
        # plt.plot(self.time, self.dtq_sum[:,60])
        # plt.plot(self.time, self.dtq_off[:,60])
        # plt.figure()
        # plt.plot(self.time, self.dtu_sum[:,40])
        # plt.plot(self.time, self.dtu_off[:,40])
        # plt.show()


    def calc_tendency(self,array, dt):
        tend = np.zeros_like(array)
        tend[1:,:] = (array[1:,:] - array[:-1,:]) / dt
        return tend

    def to_netcdf(self, file_name):

        def add_variable(file, name, type, dims, accumulated, ncatts, data):
            v = file.createVariable(name, type, dims, fill_value=nc4.default_fillvals['f4'])
            v.setncatts(ncatts)
            if accumulated:
                v.setncattr('type', 'accumulated')
            else:
                v.setncattr('type', 'instantaneous')

            if dims[-1] in ['level', 'hlevel']:
                v[:] = data[:,::-1]
            else:
                v[:] = data[:]

        # Create new NetCDF file
        f = nc4.Dataset(file_name, 'w')

        # Set some global attributes
        f.setncattr('name','Cabauw')
        f.setncattr('central_lat', self.lat)
        f.setncattr('central_lon', self.lon)
        f.setncattr('area_size', str(int((buffer*2.5)**2)) + 'km3')
        f.setncattr('Conventions', "CF-1.4")
        f.setncattr('institute_id', "KNMI")
        f.setncattr('model_id', "harmonie-40h1.2.tg2")
        f.setncattr('domain', "NETHERLANDS")
        f.setncattr('driving_model_id', "ERA5")
        f.setncattr('experiment_id', "WINS50_40h12tg2_fERA5")
        f.setncattr('title', "WINS50 - initial & boundary conditions for LES")
        f.setncattr('project_id', "WINS50")
        f.setncattr('institution', "Royal Netherlands Meteorological Institute, De Bilt, The Netherlands")
        f.setncattr('data_contact', "Natalie Theeuwes, R&D Weather & Climate Models, KNMI (natalie.theeuwes@knmi.nl)")

        # Create dimensions
        f.createDimension('time',   self.nt)
        f.createDimension('level',  self.nlev)
        # f.createDimension('levelh', self.nlevh)

        # Dimensions in NetCDF file
        dim3d  = ('time', 'level')
        # dim3dh = ('time', 'levelh')
        dim2d  = ('time')
        dim1d  = ('time')

        # Output data type
        dtype = 'f4'

        # # Domain information
        # # name        = f.createVariable('name',        str,  ('domain')); name[:] =  'Cabauw'  # change 
        # central_lat = f.createVariable('central_lat', 'f4', ); central_lat = lat
        # central_lon = f.createVariable('central_lon', 'f4', ); central_lon = lon
        # west_lon    = f.createVariable('west_lon',    'f4', ); west_lon    = None
        # east_lon    = f.createVariable('east_lon',    'f4', ); east_lon    = None
        # north_lat   = f.createVariable('north_lat',   'f4', ); north_lat   = None
        # south_lat   = f.createVariable('south_lat',   'f4', ); south_lat   = None

        # Create spatial/time variables
        add_variable(f, 'time', dtype, dim1d,  False, {'units': 'hours since 2010-01-01 00:00:00', 'long_name': 'time', 'calender': 'standard'}, self.hours_since)
        add_variable(f, 'z',    dtype, dim3d,  False, {'units': 'm',  'long_name': 'Full level geopotential height'}, self.z)
        add_variable(f, 'p',    dtype, dim3d,  False, {'units': 'Pa', 'long_name': 'Full level hydrostatic pressure'}, self.p)
        # add_variable(f, 'zh',   dtype, dim3dh, False, {'units': 'm',  'long_name': 'Half level geopotential height'}, self.zh)
        # add_variable(f, 'ph',   dtype, dim3dh, False, {'units': 'Pa', 'long_name': 'Half level hydrostatic pressure'}, self.ph)

        # Model variables
        add_variable(f, 'T',    dtype, dim3d, False, {'units': 'K',       'long_name': 'Absolute temperature'}, self.T)
        add_variable(f, 'u',    dtype, dim3d, False, {'units': 'm s-1',   'long_name': 'Zonal wind'}, self.u)
        add_variable(f, 'v',    dtype, dim3d, False, {'units': 'm s-1',   'long_name': 'Meridional wind'}, self.v)
        add_variable(f, 'q',    dtype, dim3d, False, {'units': 'kg kg-1', 'long_name': 'Total specific humidity'}, self.q)
        add_variable(f, 'qi',    dtype, dim3d, False, {'units': 'kg kg-1', 'long_name': 'Specific humidity (ice)'}, self.qci)
        add_variable(f, 'ql',    dtype, dim3d, False, {'units': 'kg kg-1', 'long_name': 'Specific humidity (liquid)'}, self.qcw)

        # # Net radiative fluxes
        # add_variable(f, 'sw_net', dtype, dim3dh, True, {'units': 'W m-2', 'long_name': 'Net shortwave radiation'}, self.sw_rad)
        # add_variable(f, 'lw_net', dtype, dim3dh, True, {'units': 'W m-2', 'long_name': 'Net longwave radiation'}, self.lw_rad)

        # Surface variables
        add_variable(f, 'H',       dtype, dim2d, True,  {'units': 'W m-2',   'long_name': 'Surface sensible heat flux'}, self.H)
        add_variable(f, 'LE',      dtype, dim2d, True,  {'units': 'W m-2',   'long_name': 'Surface latent heat flux'}, self.LE)
        add_variable(f, 'T_s',     dtype, dim2d, False, {'units': 'K',       'long_name': 'Absolute (sea) surface temperature'}, self.Tsk)
        add_variable(f, 'q_s',     dtype, dim2d, False, {'units': 'kg kg-1', 'long_name': 'Surface specific humidity'}, self.q[:,-1])
        add_variable(f, 'p_s',     dtype, dim2d, False, {'units': 'Pa',      'long_name': 'Surface pressure'}, self.ps)
        # add_variable(f, 'lwin_s',  dtype, dim2d, True,  {'units': 'W m-2',   'long_name': 'Surface shortwave incoming radiation'}, self.lwds)
        # add_variable(f, 'swin_s',  dtype, dim2d, True,  {'units': 'W m-2',   'long_name': 'Surface longwave incoming radiation'}, self.swds)

        # # Soil variables
        # if self.add_soil:
        #     add_variable(f, 'Tg1',     dtype, dim2d, False, {'units': 'K',       'long_name': 'Top soil layer temperature'}, self.tg1)
        #     add_variable(f, 'Tg2',     dtype, dim2d, False, {'units': 'K',       'long_name': 'Bulk soil layer temperature'}, self.tg2)
        #     add_variable(f, 'wg1',     dtype, dim2d, False, {'units': 'm3 m-3',  'long_name': 'Top soil layer moisture content'}, self.wg1)
        #     add_variable(f, 'wg2',     dtype, dim2d, False, {'units': 'm3 m-3',  'long_name': 'Bulk soil layer moisture content'}, self.wg2)
        #     add_variable(f, 'wg3',     dtype, dim2d, False, {'units': 'm3 m-3',  'long_name': 'Bottom soil layer moisture content'}, self.wg3)

        # for qtype,qname in self.qtypes.items():
        #     add_variable(f, qtype, dtype, dim3d, False, {'units': 'kg kg-1', 'long_name': 'Specific humidity ({})'.format(qname)}, getattr(self, qtype))

        # Tendencies
        add_variable(f, 'dtT_phy', dtype, dim3d, True, {'units': 'K s-1',  'long_name': 'Physics temperature tendency'},  self.dtT_phy)
        add_variable(f, 'dtT_dyn', dtype, dim3d, True, {'units': 'K s-1',  'long_name': 'Dynamics temperature tendency'}, self.dtT_dyn)
        # add_variable(f, 'dtT_rad', dtype, dim3d, True, {'units': 'K s-1',  'long_name': 'Radiative temperature tendency'}, self.dtT_rad)

        add_variable(f, 'dtu_phy', dtype, dim3d, True, {'units': 'm s-2',  'long_name': 'Physics zonal wind tendency'},  self.dtu_phy)
        add_variable(f, 'dtu_dyn', dtype, dim3d, True, {'units': 'm s-2',  'long_name': 'Dynamics zonal wind tendency'}, self.dtu_dyn)

        add_variable(f, 'dtv_phy', dtype, dim3d, True, {'units': 'm s-2',  'long_name': 'Physics meridional wind tendency'},  self.dtv_phy)
        add_variable(f, 'dtv_dyn', dtype, dim3d, True, {'units': 'm s-2',  'long_name': 'Dynamics meridional wind tendency'}, self.dtv_dyn)

        add_variable(f, 'dtq_phy', dtype, dim3d, True, {'units': 'kg kg-1 s-1',  'long_name': 'Physics total specific humidity tendency'},  self.dtq_phy)
        add_variable(f, 'dtq_dyn', dtype, dim3d, True, {'units': 'kg kg-1 s-1',  'long_name': 'Dynamics total specific humidity tendency'}, self.dtq_dyn)

        add_variable(f, 'dtqc_phy', dtype, dim3d, True, {'units': 'kg kg-1 s-1',  'long_name': 'Physics total cloud (water + ice) mixing ratio tendency'},  self.dtqc_phy)
        add_variable(f, 'dtqc_dyn', dtype, dim3d, True, {'units': 'kg kg-1 s-1',  'long_name': 'Dynamics total  cloud (water + ice) mixing ratio tendency'}, self.dtqc_dyn)

        f.close()


if __name__ == '__main__':

    dt = 60 # model timestep
    step = 60 # output 
    lat_select = 51.971  # Cabauw 
    lon_select = 4.927   # Cabauw
    buffer = 3           # buffer of 15 km around (7.5 km on each side) the gridpoint 3 * 2 * 2.5 km
    start_date = datetime(2020,9,1)
    last_date = datetime(2020,9,2)
    exp_name = 'ruisdael_IOP2020'
    datadir = ''

    date_array = [start_date + timedelta(days = x) for x in range(0,(last_date-start_date).days + 1)]
    
    for date in date_array:
        print("starting scripts for {0}".format(date))
        get_from_ecfs_daily(date)
        get_from_ecfs_monthly(date)
   
        data = harmonie_to_LESforce(datadir, exp_name, date, lat_select, lon_select, buffer = 3, add_soil=False, dt=step*dt, quiet=False)

        data.to_netcdf('LES_forcing_{0:04d}{1:02d}{2:02d}.nc'.format(date.year,date.month,date.day))


