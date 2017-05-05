#!/usr/bin/env python
# Mark Richardson, CEMAC
# adapted from work by Robin Steven at SEE Leeds

# deter use of pylab
from pylab import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import netCDF4 as nc4

# an exit function
def exit_four (exit_string):
  "An error has occirred:"
  print exit_string
  # how to exit?

# a function to count how many time step s to a particular date
def getStart(start_year,start_month,start_date) :
  ''' Countthe number of days from start of file to the supplied date.

      arg list is integers
  '''
  dpm = [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]
  month_str = ['January', 'February', 'March', 'April', 'May', 'June',
               'July','August','September','October','November','December'  ]
  tsteps = 0
  for YYYY in np.arange(2000,start_year+1,1):
    ###print "DBG year:", YYYY
    for MM in np.arange(1,start_month,1):
      ###print "DBG: month ", MM
      for DD in np.arange ( 1, dpm[MM-1]+1 ):
        ###print "DBG: date ", DD, month_str[MM-1]
        date_str = str("%04d"%YYYY) + str("%02d"%MM) + str("%02d"%DD)
        for hh in np.arange(1,25,1):
          # each hour
          tsteps = tsteps + 1
  print "Day before requested start is ",date_str

  # end year was YYYY perhaps need to examine how multi year will run
  MM = MM+1
  for DD in np.arange ( 1, start_date+1 ): # can chose any date within the month
    print "DBG: extend date ", DD
    date_str = str("%04d"%YYYY) + str("%02d"%MM) + str("%02d"%DD)
    for hhhh in np.arange(1,24,1):
      # each hour
      tsteps = tsteps + 1
  endday   = DD
  date_str = str("%04d"%YYYY) + str("%02d"%MM) + str("%02d"%DD)
  print "Established start date:" , date_str, "is ",tsteps," from beginning of file"
  return tsteps

# --- PARAMETERS ---
# Should check if the basemap exists
try:
    from mpl_toolkits.basemap import Basemap
    dobasemap = True                     
except ImportError:                         
    dobasemap = False                    

# MAIN ####
# set a destination for the plots
###PngDir="/nfs/see-fs-02_users/earmgr/tmpOmniGlobeContent/"
# Terminate directory name with forward slash /
PngDir="/nfs/ncas/earmgr/CESM_OmniGlobe/FullHD/"
# Need to choose a file
file_a='base2000_cesm111_v2_hourly_avg.nc'
# this is the reference topology JPG
etopo_jpg = '/nfs/see-fs-02_users/earmgr/Images/etopo3000.jpg'
big_pos = 1.0e100
big_neg = -1.0e100
# parse cmd line for choice of variable to inspect
# O3_SRF, ISOP_SRF (but T is 3D so need some filtering to get surface T)
VarChoice = 1

# do we want a daylight indication
req_lightlevel = False
# Target visualisation system
OmniGlobe=False
FullHD = True 
Mon1680 = False
# The number of pixels in the output is related to the "inches" and dots per inch
# this will be set by user in production code
# 100 dpi seems to be default save image setting
# So values are inches to get a set number of pixels 
# when doing FullHD need to think about 80dpi (how to set, why to set?)
if OmniGlobe:
  h_inches = 30.0  # 3000 by 1500 i.e. OmniGlobe at 100dpi
  v_inches = 15.0
elif FullHD:
  h_inches = 19.2   # 1920 by 1080 i.e. FullHD@100dpi
  v_inches =  9.6   # to retain 2:1 aspect ratio of image
  ###v_inches = 10.8
  #h_inches = 24.0 v_inches = 13.5 # 1920 by 1080 i.e. FullHD@80dpi
elif Mon1680:
  h_inches = 16.8   # 1680x1050@100dpi
  v_inches = 10.5

# --- INITIALIZE ---
lightlevel = np.zeros([96,145])  # CAUTION size has been hardwired
#
# sunlight level 0.0 = no sunlight, 1.0 is maximum sunlight (tbd)
night = 0.0
light = 1.0

# At midnight 0/01/2001, adjust these per hour (shift by 6 cells to left - subtracting)
# AGAIN CAUTION DUE TO HARDWIRED SIZE
refdusk = 108  
refdawn = 36
CellsPerDay = 144
hr_shift = CellsPerDay/24   # for different resolution will auto adjust to one hour shift

# this is a mechanism for generating light and dark data
lux = np.linspace(0,144,145,endpoint=True)   # make sure lux is a list
# turn on the light (there is a more efficient method for this I am sure
for i in np.arange(CellsPerDay+1):
  lux[i] = light         # cells 0 to CellsPerDay (+1 for faces )

# Should we allow for choosing different variables within the code or preset it?
if VarChoice == 1:
  ChoiceOfVar = 'O3_SRF'
  VarName = "o3_surf_"
elif VarChoice == 2 :
  ChoiceOfVar = 'ISOP_SRF'
  VarName = "isop_s_"
elif VarChoice == 3 :
  ChoiceOfVar = 'T'
  VarName = "degK_"

if dobasemap:
  print "Creating canvas basemap to draw on"
  ###bm1 = Basemap(resolution='c', llcrnrlon=0.0,llcrnrlat=-90, urcrnrlon=360.0,urcrnrlat=90)
  bm1 = Basemap(resolution='c', llcrnrlon=-180.0,llcrnrlat=-90, urcrnrlon=180.0,urcrnrlat=90)

##### OUR DATA for displaying ###############
ncfile_a = nc4.Dataset(file_a, 'r')

# Extract latitude, longitude
lons = ncfile_a.variables['lon'][:]
lats = ncfile_a.variables['lat'][:]

# Cater for lack of 360.0 information (periodic)
nlat = len(lats)
nlon = len(lons)
meridian = nlon/2
print "INFO: nlon,meridian and nlat are:", nlon,meridian,nlat
# need to add one to lons and also convert to -180 to 180 range
nlon_p = nlon+1
lons_p = np.zeros(nlon_p)
lons_p[0] = -180.0
on2 = 1
for on in np.arange(meridian+1,nlon):
  lons_p[on2] = lons[on]-360.0
  on2 = on2 +1
for on in np.arange(0,meridian+1):
  lons_p[on2] = lons[on]
  on2 = on2 +1
# end column of lats

# create 2D x and 2D y arrays
nlon_m = nlon_p
xm = zeros([nlat,nlon_m])
ym = zeros([nlat,nlon_m])
if dobasemap:
  xm,ym = bm1(*np.meshgrid(lons_p,lats))
else:
  xm,ym = np.meshgrid(lons_p,lats)

# Make a copy of colormap and make it scale from transparent to solid
c_map = plt.cm.Reds
b_map = plt.cm.Blues
m_map = c_map(np.arange(c_map.N) )
m_map[:,-1] = np.linspace(0,1,c_map.N)
m_map = ListedColormap(m_map)

# Assume lats lons persist - avoid reading them repeatedly
# For each timeslice of interest (established in the loop limits and stride)
start = getStart(2000,8,1)  # 1st august 2000 (now count from start of file)
if start == -99 :
  exit_msg = "Error in calculating start, check date requested."
  exit_four(exit_msg)

# TODO make this method better
datestart = 20000801    # use the integer counter with in a month
finish = start + (30*24)  # 30 days
###finish = start + 1  # 1 frame only in dev code
stride = 1

# Lookahead to determine min max over full sim
###mriprint "lookahead"
###mrigmin = big_pos
###mrigmax = big_neg
###mrifor simhour in np.arange(start,finish,stride):
###mri  # Extract chosen var surface dat (2D geom)
###mri  chosen_a = ncfile_a.variables[ChoiceOfVar][simhour,:,:]
###mri  lmin = np.amin(chosen_a)
###mri  gmin = np.min(gmin,lmin)
###mri  lmax = np.amax(chosen_a)
###mri  gmax = np.max(gmax,lmax)
###mri  print "DBG: lmin, lmax:",lmin,lmax, chosen_a.shape
###mriprint "Looked ahead for an overall Min,Max", gmin,gmax
###mri
datestamp = "01 August 2000"
timestamp = "00:00"

chosen_p = np.zeros( [nlat, nlon+1] ) # P the plottable version has extra column
print "Generating images ",start, datestamp, timestamp
for simhour in np.arange(start,finish,stride):
  # Extract chosen var surface dat (2D geom)
  chosen_a = ncfile_a.variables[ChoiceOfVar][simhour,:,:]
  
  for at in np.arange(0,nlat):
    chosen_p[at][0] = chosen_a[at][meridian]
    on2 = 1
    for on in np.arange(meridian+1,nlon):
       chosen_p[at][on2] = chosen_a[at][on]
       on2 = on2 +1
    for on in np.arange(0,meridian+1):
       chosen_p[at][on2] = chosen_a[at][on]
       on2 = on2 +1

  if req_lightlevel:
    # Need an artificial sunlight representation to sync with "simhour"
    # reference where the terminators exist (naive north-south line)
    dlight = mod(simhour,24)  # number of hours from midnight, requires first dataset to midnight
    # Dusk
    dusk=fmod((refdusk-dlight*hr_shift),CellsPerDay)
    if dusk < 0 : 
      dusk = CellsPerDay + dusk
    # Dawn
    dawn=fmod((refdawn-dlight*hr_shift),CellsPerDay)
    if dawn < 0 : 
      dawn = CellsPerDay + dawn
    
    ###print 'Dlight=',dlight,'Dusk=',dusk,' Dawn=',dawn
    # between dusk and dawn it is dark
    if (dusk < dawn):
      lux[0:dusk] = light
      lux[dusk:dawn] = night
      lux[dawn:CellsPerDay] = light
    else:
      lux[0:dawn] = night
      lux[dawn:dusk] = light
      lux[dusk:CellsPerDay] = night
    
    # now propogate over all latitudes (96 in this specific sim)
    for j in np.arange(nlat):
      for i in np.arange(CellsPerDay+1):
        lightlevel[j][i] = lux[i]
  # END OF setting up data now do the figure planning
  
  # the inches were determined near start of script
  plt.figure(figsize=(h_inches,v_inches))
  
  # make canvas as big as figure area (for OmniGlobe PNGs)
  plt.subplots_adjust(0,0,1,1)
  
  if dobasemap:
    # draw coastlines, country boundaries, fill continents.
    bm1.drawcoastlines(linewidth=0.25)
    # draw the edge of the map projection region (the projection limb)
    ###bm1.drawmapboundary()
    # draw lat/lon grid lines every 30 degrees. But omit labels
    ###bm1.drawmeridians(np.arange(0.,360.,60.),labels=[0,0,0,0],fontsize=10,linewidth=0.25)
    bm1.drawparallels(np.arange(-90.,120.,30.),labels=[0,0,0,0],fontsize=10,linewidth=0.5)

    # now plot the quantity on tthe map
    bm1.pcolormesh(xm,ym,chosen_p,cmap=m_map)

    # indicate the night and day regions
    if req_lightlevel:
      bm1.pcolormesh(lons_p, lats, lightlevel , cmap=plt.cm.gray,alpha=0.2)

    # try put the geography under data
    ###bm1.warpimage( image=etopo_jpg )
    bm1.warpimage(image='/nfs/see-fs-02_users/earmgr/Images/etopo3000.jpg',scale=0.64)
    ###bm1.etopo(scale=0.28)
    # Labels on equator
    xst = np.zeros(3)
    yst = np.zeros(3)
    xst[0],yst[0] = bm1(65.0,0.0)
    xst[1],yst[1] = bm1(-160.0,0.0)
    xst[2],yst[2] = bm1(-45.0,0.0)

  else:
    print "Basemap not found plotting anyway"
    # Now to plot the actual data
    plt.pcolormesh(lons_p, lats, chosen_p , cmap=plt.cm.Purples)
    # indicate the night and day regions
    if req_lightlevel:
      plt.pcolormesh(lons_p, lats, lightlevel , cmap=plt.cm.gray,alpha=0.2)

  # advance the date counter
  days = (simhour-start)/24 # will be zero for first 24 hours
  timestamp = datestart+days   # first pass of the loop will be 0 days for 24 hours
  datestamp = str("%02d"%(days+1))+" August 2000 "
  current_hour = np.mod( (simhour-start),24 )
  hour_string=str("%02d:00 "%current_hour)
  zst = hour_string+datestamp
  plt.text(xst[0],yst[0],zst,size='x-large')
  plt.text(xst[1],yst[1],zst,size='x-large')
  plt.text(xst[2],yst[2],zst,size='x-large')
  
  plt.plot()    # Render the image
  
  # should do better to change this and introduce checking
  Frame = str("%04d"%simhour)
  MetaData = VarName+str(timestamp)+"_"+Frame
  fname_out = PngDir+MetaData+".png"
  
  plt.savefig(fname_out,bbox_inches="tight",pad_inches=0.0)

  # Iteration end one figure per timestep
  print "Saved figure ",fname_out
  print hour_string+datestamp
###  if simhour > finish-stride-2:
###    plt.show()                        # THIS IS NOT SENSIBLE in production
    #draw()    # not sure if this is right invocation
  plt.clf()    # clear the figure but leave the canvas
  # still trying to decide how to use next as I would like to see at least one figure
  plt.close()  # could be explicit with figure name or number

###show() # Show final figure (I hope)
# END OF SCRIPT
