import numpy as num
import sys
import os
import colorama
colorama.init(autoreset=True)
import ccdproc
from astropy import units as u
from astropy.stats import SigmaClip, mad_std


def out_error(msg):
    print(colorama.Fore.RED+"Error: %s"%msg)
    sys.exit()

def out_warning(msg):
    print(colorama.Fore.YELLOW+"Warning: %s"%msg)

def out_info(msg):
    print(colorama.Fore.GREEN+"%s"%msg)

# To read parameter file
class read_paramfile:
    def __init__(self, pname):
        print("Opening parameter file %s"%pname)
        if not os.path.exists(pname):
            out_error("File %s does not exist."%pname)
        else:
            pd = open(pname, "r")
            pardata = pd.readlines()
            pd.close()
            # Eleccion de objeto
            pos = 1
            self.target = pardata[pos+0].split()[2]         # Target
            self.tg_ra  = float(pardata[pos+1].split()[2])         # Target RA in deg
            self.tg_dec = float(pardata[pos+2].split()[2])         # Target DEC in deg
            self.tg_pmra  = float(pardata[pos+3].split()[2])       # Target PM RA in deg
            self.tg_pmdec = float(pardata[pos+4].split()[2])       # Target PM DEC in deg
            self.tg_parallax = float(pardata[pos+5].split()[2])       # Target parallax in mas            
            self.tg_rv    = float(pardata[pos+6].split()[2])       # Target mean systemic RV in km/s
            #Parametros de observacion
            pos += 8
            self.instrument = pardata[pos+0].split()[2]         # Instrumento
            self.obsdate = pardata[pos+1].split()[2]         # Fecha de observacion 
            self.selec = pardata[pos+2].split()[2]         # Calibracion aplicada
            self.nref = int(pardata[pos+3].split()[2])    # Numero de est. de referencia para fotometria
            #Parametros calibracion
            pos += 5
            self.bias = pardata[pos+0].split()[2]         # If Bias
            self.dark = pardata[pos+1].split()[2]         # If Dark
            self.flat = pardata[pos+2].split()[2]         # If Flat
            self.bpix = pardata[pos+3].split()[2]         # If BadPix
            self.dtol = float(pardata[pos+4].split()[2])  # Tolerance in seconds to accept a DARK with EXPTIME
            self.calsigclip = float(pardata[pos+5].split()[2])      # Sigma clipping in combination
            self.science_box = pardata[pos+6].split()[2:6]   # Science region on images (to exclude overscan)
            #Parametros Fotometria
            pos += 8 #22
            info_apert = pardata[pos+0].split()[2].split(',')     # Aperturas para fotometria
            self.apertures = num.arange(float(info_apert[0]), float(info_apert[1]), float(info_apert[2]))
            #info_annuli = pardata[pos+1].split()[2].split(',')    # Sky inner annulus
            #self.annuli = num.arange(float(info_annuli[0]), float(info_annuli[1]), float(info_annuli[2]))
            self.annuli = float(pardata[pos+1].split()[2])
            info_dannuli = pardata[pos+2].split()[2].split(',')   # Width of sky annulus
            #self.dannuli = num.arange(float(info_dannuli[0]), float(info_dannuli[1]), float(info_dannuli[2]))
            self.dannuli = float(pardata[pos+2].split()[2])
            #Parametros Generales
            pos += 4 #26
            self.threshold = float(pardata[pos+0].split()[2])    # Finding Threshold
            self.fwhm = float(pardata[pos+1].split()[2])         # FWHM of stars in images
            self.srad = float(pardata[pos+2].split()[2])         # Tolerance redius to find stars around selected center
            self.niter = int(pardata[pos+3].split()[2])          # Number of iterations in sigma-clip
            self.plotstep = int(pardata[pos+4].split()[2])       # Frequency of centering plots: 0 (no plots but first), 1 (plot every file), N (every N files)
            self.centering_algorithm = pardata[pos+5].split()[2] # Centering Algorithm
            self.centering_box = pardata[pos+6].split()[2]       # Centering Box
            self.sky_fit_algorithm = pardata[pos+7].split()[2]   # Sky Fitting Algorithm
            #
            #impath = "../imagenes/%s/%s/calibrated/%s/%s/"%(self.instrument,self.obsdate,self.target,self.selec)
            #listfile = impath+"images.lst"
            #self.listfile = listfile
            #self.impath = impath
            #if os.path.exists(listfile):
            #    print("Calculating photometry for files in %s"%listfile)
            #else:
            #    out_error("File %s does not exist."%listfile)

def closest_dark_id(d_dit, im_dit, tol=0.5):
    dist = num.abs(num.asarray(d_dit) - im_dit)
    id_min = num.argmin(dist)
    if (dist[id_min] > tol):
        out_error("There are no darks with exposure times closer than %.1f from the available images"%tol)
    else:
        return id_min

def find_filter_id(filters, im_filter):
    idx = num.argwhere(filters == im_filter)
    if len(idx) == 0:
        out_error("There is no flat for filter %s"%im_filter)
    else:
        return idx[0][0]

def imcombine(filenames, sigclip=3.0):
    images = ccdproc.ImageFileCollection(filenames=filenames, keywords='*')
    print(images.summary)
    #combiner = ccdproc.Combiner(images.ccds(ccd_kwargs={'unit': 'adu'}))
    #combiner = ccdproc.Combiner(images.ccds(ccd_kwargs='*'))
    #combiner.sigma_clipping(low_thresh=sigclip, high_thresh=sigclip, func=num.ma.median)
    #combined_median = combiner.median_combine()
    combined_median = ccdproc.combine(filenames, method='average', ccdkwargs='*', unit='adu', format='fits', sigma_clip=sigclip>0.0, sigma_clip_low_thresh=sigclip, sigma_clip_high_thresh=sigclip, sigma_clip_func=num.ma.median, sigma_clip_dev_func=mad_std, mem_limit=350e6)
    return combined_median


def flatcombine(filenames, bias=None, dark_list=None, dark_list_dits=None, cdir='.', filt=None, sigclip=3.0, scienceframe=[], tol=0.5):
    if bias== None and dark_list==None:
        out_error("Either BIAS or DARK should be applied.")
    fsigclip = SigmaClip(sigma=sigclip, cenfunc='median')
    if len(scienceframe) == 0:
        out_error("In flatcombine: should add scienceframe to call")
    images_collection = ccdproc.ImageFileCollection(filenames=filenames, keywords='*')
    i = 0
    if not os.path.exists(cdir):
        out_warning("Creating directory tree %s"%cdir)
        os.makedirs(cdir)
    images_collection.files_filtered(filter=filt)
    all_cflat_files = []
    for image, fname in images_collection.ccds(return_fname = True, ccd_kwargs={'unit': 'adu'}):
        if bias != None:
            image_cbias = ccdproc.subtract_bias(image, bias)
        else:
            image_cbias = image
        cflatfile = "%s/tmp_flat_%s_%d.fits"%(cdir, filt, i)
        i += 1
        data = image_cbias.data[scienceframe]
        norm = num.median(fsigclip(data))
        norm_cflat = image_cbias.divide(norm * u.adu)
        if dark_list != None:
            flat_dit = float(image.header['EXPTIME'])
            idx = closest_dark_id(dark_list_dits, flat_dit, tol=tol)
            image_cdark = ccdproc.subtract_dark(image_cbias, dark_list[idx], exposure_time="EXPTIME", exposure_unit=u.second)
            data = image_cdark.data[scienceframe]
            norm = num.median(fsigclip(data))
            norm_cflat = image_cdark.divide(norm * u.adu)
        norm_cflat.write(cflatfile, overwrite=True)
        all_cflat_files.append(cflatfile)
    cflat = imcombine(all_cflat_files, sigclip=sigclip)
    return cflat


#def return_instrument(telescope, instrument):
    
class define_instrument:
    def __init__(self, telescope):
        if telescope == "TRAPPIST":
            self.gain = 1.0 # e-/adu
            self.ron = 14.0   # e-
            self.obs_lat  = -29.256666
            self.obs_long = -70.73
            self.obs_elev = 2347.0
            self.obs = 'lasilla'
        elif telescope == "SPECULOOS1":
            self.gain = 1.04 # e-/adu
            self.ron = 6.2   # e-
            self.obs_lat  = -24.625278
            self.obs_long = - 70.402222
            self.obs_elev = 2635.0
            self.obs = 'paranal'    
        elif telescope == "SPECULOOS2":
            self.gain = 1.04 # e-/adu
            self.ron = 6.2   # e-
            self.obs_lat  = -24.625278
            self.obs_long = - 70.402222
            self.obs_elev = 2635.0
            self.obs = 'paranal'    
        elif telescope == "SPECULOOS3":
            self.gain = 1.04 # e-/adu
            self.ron = 6.2   # e-
            self.obs_lat  = -24.625278
            self.obs_long = - 70.402222
            self.obs_elev = 2635.0
            self.obs = 'paranal'    
        elif telescope == "SPECULOOS4":
            self.gain = 1.04 # e-/adu
            self.ron = 6.2   # e-
            self.obs_lat  = -24.625278
            self.obs_long = - 70.402222
            self.obs_elev = 2635.0
            self.obs = 'paranal'  
        elif telescope == 'Zeiss':
            self.gain = 1.5
            self.ron = 6.3
            self.obs_lat  = -22.535
            self.obs_long = -45.5833
            self.obs_elev = 1870.0
            self.obs = 'lna'
        elif telescope == 'LCOGT-0m4c':
            self.gain = 1.0 # e/ADU
            self.ron =  3.25 # e/pix
            self.obs_lat = 20.7069694
            self.obs_long = -156.2575333
            self.obs_elev = 3037.0
            self.obs = 'haleakala'
        else:
            out_error("Telescope %s not recognized"%telescope)

