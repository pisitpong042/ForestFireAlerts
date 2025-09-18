#process.py
try:
    from fwi import bui, dc, dmc, isi, ffmc
except:
    pass
import netCDF4
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from threading import Thread

time_total = time.time()
time_start = time.time()
domain_high = 699
domain_width = 543

GB_Lat = np.empty((0,0))
GB_Lon = np.empty((0,0))
GB_Temp_noon = np.empty((0,0))
GB_Rh_noon = np.empty((0,0))
GB_u_10m_gr_noon = np.empty((0,0))
GB_v_10m_gr_noon = np.empty((0,0))
GB_Wsp_noon = np.empty((0,0))
GB_precip_24hr = np.empty((0,0))

GB_ffmc = np.empty((domain_high,domain_width))
GB_ffmc_yda = np.empty((domain_high,domain_width))

GB_dmc = np.empty((domain_high,domain_width))
GB_dmc_yda = np.empty((domain_high,domain_width))

GB_dc = np.empty((domain_high,domain_width))
GB_dc_yda = np.empty((domain_high,domain_width))

GB_isi = np.empty((domain_high,domain_width))
GB_bui = np.empty((domain_high,domain_width))
GB_fwi = np.empty((domain_high,domain_width))
####################################################################################################
def read_nc(yesterday_nc_file, today_nc_file):
    global GB_Lat, GB_Lon, GB_Temp_noon, GB_Rh_noon, GB_Wsp_noon, GB_precip_24hr
    global GB_u_10m_gr_noon, GB_v_10m_gr_noon

    yesterday_nc = netCDF4.Dataset(yesterday_nc_file)
    precip_hr = yesterday_nc.variables['precip_hr']# <class 'netCDF4.Variable'>    float32 precip_hr(time, south_north, west_east) current shape = (169, 183, 399)
    GB_precip_24hr = np.array(precip_hr[7,:,:]) # 14.00
    times = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 , 22, 23] # 15.00 - 06.00
    for time in times: GB_precip_24hr = np.add(GB_precip_24hr,precip_hr[time,:,:])
    yesterday_nc.close()

    today_nc = netCDF4.Dataset(today_nc_file)
    ### USE UTC TIME ###
    lat = today_nc.variables['lat']            # <class 'netCDF4.Variable'>    float32 lat(south_north, west_east)         current shape = (183, 399)
    lon = today_nc.variables['lon']            # <class 'netCDF4.Variable'>    float32 lon(south_north, west_east)         current shape = (183, 399)
    t_2m = today_nc.variables['T_2m']          # <class 'netCDF4.Variable'>    float32 T_2m(time, south_north, west_east)  current shape = (169, 183, 399)
    rh_2m = today_nc.variables['rh_2m']        # <class 'netCDF4.Variable'>    float32 rh_2m(time, south_north, west_east) current shape = (169, 183, 399)
    precip_hr = today_nc.variables['precip_hr']# <class 'netCDF4.Variable'>    float32 precip_hr(time, south_north, west_east) current shape = (169, 183, 399)
    ### u_10m_gr == u-Component of Wind at 10 m (grid) eastward_wind
    ### v_10m_gr == v-Component of Wind at 10 m (grid) northward_wind
    u_10m_gr = today_nc.variables['u_10m_gr']  # <class 'netCDF4.Variable'>    float32 u_10m_gr(time, south_north, west_east) current shape = (169, 183, 399)
    v_10m_gr = today_nc.variables['v_10m_gr']  # <class 'netCDF4.Variable'>    float32 v_10m_gr(time, south_north, west_east) current shape = (169, 183, 399)

    '''
    2D Array
    south_north === row
    west_east === colum

    [[south_north, west_east, west_east,]
    [south_north,  west_east, west_east,]]
    '''
    GB_Lat = np.array(lat[:])
    GB_Lon = np.array(lon[:])
    GB_Temp_noon = np.array(t_2m[6,:,:])
    GB_Rh_noon = np.array(rh_2m[6,:,:])
    GB_u_10m_gr_noon = np.array(u_10m_gr[6,:,:])
    GB_v_10m_gr_noon = np.array(v_10m_gr[6,:,:])
    times = [0, 1, 2, 3, 4, 5] # 07.00 - 12.00
    for time in times: GB_precip_24hr = np.add(GB_precip_24hr,precip_hr[time,:,:])
    today_nc.close()

    u_10m_gr_noon_pow2 = np.power(GB_u_10m_gr_noon,2)
    v_10m_gr_noon_pow2 = np.power(GB_v_10m_gr_noon,2)
    uv_10m_gr_noon_pow2 = np.add(u_10m_gr_noon_pow2,v_10m_gr_noon_pow2)
    GB_Wsp_noon_mps = np.square(uv_10m_gr_noon_pow2) # meter per sec
    GB_Wsp_noon = np.multiply(GB_Wsp_noon_mps, 3.6) #km per hour


def cal_ffmc():
    global GB_ffmc, time_start
    for row in range(domain_high) :
        for col in range(domain_width) :
            cal = ffmc(
                GB_ffmc_yda[row][col], #ffmc_yda
                GB_Temp_noon[row][col], #temp
                GB_Rh_noon[row][col], #rh
                GB_Wsp_noon[row][col], #ws
                GB_precip_24hr[row][col] #prec
            )
            GB_ffmc[row][col] = cal
    print("FFMC Total Process Time ", round(time.time() - time_start,2), " Second")
    time_start = time.time()


def cal_dmc():
    global GB_dmc, time_start
    for row in range(domain_high) :
        for col in range(domain_width) :
            cal = dmc(
                GB_dmc_yda[row][col], #dmc_yda
                GB_Temp_noon[row][col], #temp
                GB_Rh_noon[row][col], #rh
                GB_precip_24hr[row][col], #prec
                15, #lat
                1 #mon
            )
            GB_dmc[row][col] = cal
    print("DMC Total Process Time ", round(time.time() - time_start,2), " Second")
    time_start = time.time()


def cal_dc():
    global GB_dc, time_start
    for row in range(domain_high) :
        for col in range(domain_width) :
            cal = dc(
                GB_dc_yda[row][col], #dmc_yda
                GB_Temp_noon[row][col], #temp
                GB_Rh_noon[row][col], #rh
                GB_precip_24hr[row][col], #prec
                15, #lat
                1 #mon
            )
            GB_dc[row][col] = cal
    print("DC Total Process Time ", round(time.time() - time_start,2), " Second")
    time_start = time.time()


def cal_isi():
    global GB_isi, time_start
    for row in range(domain_high) :
        for col in range(domain_width) :
            cal = isi(
                GB_ffmc[row][col], #ffmc
                GB_Wsp_noon[row][col], #ws
                True
            )
            GB_isi[row][col] = cal
    print("ISI Total Process Time ", round(time.time() - time_start,2), " Second")
    time_start = time.time()


def cal_bui():
    global GB_bui, time_start
    for row in range(domain_high) :
        for col in range(domain_width) :
            cal = bui(
                GB_dmc[row][col], #ffmc
                GB_dc[row][col] #ws
            )
            GB_bui[row][col] = cal
    print("BUI Total Process Time ", round(time.time() - time_start,2), " Second")
    time_start = time.time()


def cal_fwi():
    global GB_fwi
    for row in range(domain_high) :
        for col in range(domain_width) :
            cal = bui(
                GB_isi[row][col], #ffmc
                GB_bui[row][col] #ws
            )
            GB_fwi[row][col] = cal
    print("FWI Total Process Time ", round(time.time() - time_start,2), " Second")
    print("=== Total Process Time ", round(time.time() - time_total,2), " Second")

def create_dc_graph(ticks = [], name='redwhite.png'):
    #ticks has to be an arry of 5 values
    cmap = colors.ListedColormap(['blue','green','yellow','red','brown'])
    bounds = ticks
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Count cells in each category
    category_counts = []
    for i in range(len(ticks) - 1):
        count = np.sum((GB_dc >= ticks[i]) & (GB_dc < ticks[i+1]))
        category_counts.append(count)
    # Add count for the last category (>= last tick)
    count = np.sum(GB_dc >= ticks[-1])
    category_counts.append(count)

    # Print the counts
    print("Cell counts per category:")
    for i in range(len(ticks) - 1):
        print(f"Category {i+1} ({ticks[i]} to {ticks[i+1]}): {category_counts[i]}")
    print(f"Category {len(ticks)} (>={ticks[-1]}): {category_counts[-1]}")


    #Update Sep 12 marko: visualize DC with ticks from the slide sent by Chai
    img = plt.imshow(GB_dc,interpolation='nearest', origin='lower',cmap=cmap, norm=norm)
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=ticks)
    plt.savefig(name)
    plt.show() # this shows the plot
####################################################################################################

if __name__ == "__main__":
    # Get a list of all files in the current directory
    all_files = os.listdir('.')

    # Filter for files ending with 'd03.nc' and sort them by modification time
    target_files = sorted([f for f in all_files if f.endswith('d03.nc')], key=os.path.getmtime, reverse=False)

    # Select the bottommost two files
    bottommost_files = []
    if len(target_files) >= 2:
        bottommost_files = target_files[-2:] # Select the last two files
        print(f"The bottommost two files ending with '00UTC_d03.nc' are: {bottommost_files}")
    elif len(target_files) == 1:
        bottommost_files = target_files[-1:] # Select the last file
        print(f"Only one file ending with '00UTC_d03.nc' found: {bottommost_files}")
    else:
        print("No files ending with '00UTC_d03.nc' found.")

    print(bottommost_files)

    yesterday = bottommost_files[0].split('_')[0]
    today = bottommost_files[1].split('_')[0]

    read_nc('{}_00UTC_d03.nc'.format(yesterday),'{}_00UTC_d03.nc'.format(today))

    #GB_ffmc_yda = np.load('index_yesterday/{}_ffmc.npy'.format(yeaterday)) # load
    #GB_dmc_yda = np.load('index_yesterday/{}_dmc.npy'.format(yeaterday)) # load
    #GB_dc_yda = np.load('index_yesterday/{}_dc.npy'.format(yeaterday)) # load

    thread_01 = Thread(target=cal_ffmc,args=())
    thread_01.start()
    thread_01.join()

    thread_02 = Thread(target=cal_isi,args=())
    thread_02.start()
    thread_02.join()

    thread_03 = Thread(target=cal_dmc,args=())
    thread_03.start()
    thread_03.join()

    thread_04 = Thread(target=cal_dc,args=())
    thread_04.start()
    thread_04.join()

    thread_05 = Thread(target=cal_bui,args=())
    thread_05.start()
    thread_05.join()

    thread_06 = Thread(target=cal_fwi,args=())
    thread_06.start()
    thread_06.join()


    np.save('index_yesterday/{}_ffmc.npy'.format(today), GB_ffmc) # save
    np.save('index_yesterday/{}_dmc.npy'.format(today), GB_dmc) # save
    np.save('index_yesterday/{}_dc.npy'.format(today), GB_dc) # save


    """ in this dataset each component will be
    in the form nt,nz,ny,nx i.e. all the variables will be flipped. """
    #plt.imshow(GB_ffmc)
    ticks = [0,10,334,451,600] #from thresholds
    create_dc_graph(ticks, 'redwhite.png')
