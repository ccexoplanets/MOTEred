import numpy as num
import matplotlib.pyplot as plt
import sys
import os
import glob
import shutil
from astropy.nddata import CCDData
from astropy import units as u
import ccdproc
from cc_transits import *

if len(sys.argv) != 2:
    out_error("Use: python photometry.py PARFILE.par")
else:
    parfile = sys.argv[1]
    par = read_paramfile(parfile)

target     = par.target
instrument = par.instrument
obsdate    = par.obsdate
selec      = par.selec
csigclip   = float(par.calsigclip)
scibox     = par.science_box
scibox     = [int(x) for x in scibox]
inst = define_instrument(instrument)
instrument = par.instrument
gain = inst.gain
ron = inst.ron

# Estructura de directorios
dir_imgs = "../imagenes/%s/%s"%(instrument,obsdate)
dir_raw  = "%s/raw"%(dir_imgs)
dir_cal  = "%s/calibrated/%s/%s"%(dir_imgs, target, selec)
dir_bias = "%s/bias"%(dir_raw)
dir_dark = "%s/dark"%(dir_raw)
dir_flat = "%s/flat"%(dir_raw)
dir_bpix = "%s/badpix"%(dir_raw)
dir_rsci = "%s/science"%(dir_raw)
BIAS = par.bias == "True"
DARK = par.dark == "True"
FLAT = par.flat == "True"
BPIX = par.bpix == "True"
DTOL = par.dtol
list_bias = dir_bias+"/bias_files.lst"
list_dark = dir_dark+"/dark_files.lst"
list_flat = dir_flat+"/flat_files.lst"
list_bpix = dir_bpix+"/bpix_files.lst"
list_rsci = dir_rsci+"/science_files.lst"

cal_science_dir  = "%s/science"%(dir_cal)
cal_calibs_dir  = "%s/calibration"%(dir_cal)

if os.path.exists(cal_calibs_dir):
    out_error("Calibration directory %s already exists. Can't overwrite existing calibration %s to target %s."%(dir_cal, selec, target))

if not os.path.exists(cal_calibs_dir):
    out_info("Creating directory tree %s"%cal_calibs_dir)
    os.makedirs(cal_calibs_dir)

shutil.copy2(parfile, dir_cal)
    
if BIAS:
    if os.path.exists(dir_bias):
        if os.path.exists(list_bias):
            print("Reading file %s"%list_bias)
            fd = open(list_bias, "r")
            files_bias = fd.read().splitlines()
            fd.close()
            files_bias = ["%s/%s"%(dir_bias,files_bias[i]) for i in range(len(files_bias))]
        else:
            files_bias = glob.glob("%s/*.fits"%dir_bias)
            if len(files_bias) == 0:
                files_bias = glob.glob("%s/*.fts"%dir_bias)
                if len(files_bias) == 0:
                    out_error("Not possible to apply bias correction. No files in directory %s"%dir_bias)
    else:
        out_error("Not possible to apply bias correction. Directory %s does not exist!"%dir_bias)
if DARK:
    if os.path.exists(dir_dark):
        if os.path.exists(list_dark):
            print("Reading file %s"%list_dark)
            fd = open(list_dark, "r")
            files_dark = fd.read().splitlines()
            fd.close()
        else:
            files_dark = glob.glob("%s/*.fits"%dir_dark)
            if len(files_dark) == 0:
                files_dark = glob.glob("%s/*.fts"%dir_dark)
                if len(files_dark) == 0:
                    out_error("Not possible to apply dark correction. No files in directory %s"%dir_dark)
    else:
        out_error("Not possible to apply dark correction. Directory %s does not exist!"%dir_dark)
if FLAT:
    if os.path.exists(dir_flat):
        if os.path.exists(list_flat):
            print("Reading file %s"%list_flat)
            fd = open(list_flat, "r")
            files_flat = fd.read().splitlines()
            fd.close()
        else:
            files_flat = glob.glob("%s/*.fits"%dir_flat)
            if len(files_flat) == 0:
                files_flat = glob.glob("%s/*.fts"%dir_flat)
                if len(files_flat) == 0:
                    out_error("Not possible to apply flat correction. No files in directory %s"%dir_flat)
    else:
        out_error("Not possible to apply flat correction. Directory %s does not exist!"%dir_flat)
if BPIX:
    if os.path.exists(dir_bpix):
        if os.path.exists(list_bpix):
            print("Reading file %s"%list_bpix)
            fd = open(list_bpix, "r")
            files_bpix = fd.read().splitlines()
            fd.close()
        else:
            files_bpix = glob.glob("%s/*.fits"%dir_bpix)
            if len(files_bpix) == 0:
                files_bpix = glob.glob("%s/*.fts"%dir_bpix)
                if len(files_bpix) == 0:
                    out_warning("Not possible to read bad pixel correction file. No files in directory %s"%dir_bpix)
    else:
        out_warning("Not possible to apply bad pixel correction. Directory %s does not exist!"%dir_bpix)


print("Classifying science images")
if os.path.exists(dir_rsci):
    if os.path.exists(list_rsci):
        print("Reading file %s"%list_rsci)
        fd = open(list_rsci, "r")
        files_rsci = fd.read().splitlines()
        fd.close()
        files_rsci = ["%s/%s"%(dir_rsci,files_rsci[i]) for i in range(len(files_rsci))]
    else:
        out_warning("File %s not found. Reading ALL images in directory %s."%(list_rsci, dir_rsci))
        files_rsci = glob.glob("%s/*.fits"%dir_rsci)
        if len(files_rsci) == 0:
            files_rsci = glob.glob("%s/*.fts"%dir_rsci)
            if len(files_rsci) == 0:
                out_error("Not science files in directory %s!"%dir_rsci)
else:
    out_error("Not possible to calibrate any data. Science directory %s does not exist!"%dir_rsci)

# Leer DITS de imagenes de ciencia
rsci_images = ccdproc.ImageFileCollection(filenames=files_rsci, keywords='*')
rsci_dits = []
rsci_filters = []
rsci_NXs = []
rsci_NYs = []
rsci_targets = []
for hdu, fname in rsci_images.headers(return_fname=True):
    rsci_dits.append(hdu['EXPTIME'])
    rsci_filters.append(hdu['FILTER'])
    rsci_NXs.append(hdu['NAXIS1'])
    rsci_NYs.append(hdu['NAXIS2'])
    rsci_targets.append(hdu['OBJECT'])
unique_rsci_dits = num.unique(rsci_dits)
unique_rsci_filters = num.unique(rsci_filters)
unique_rsci_NXs = num.unique(rsci_NXs)
unique_rsci_NYs = num.unique(rsci_NYs)
unique_rsci_targets = num.unique(rsci_targets)
if len(unique_rsci_NXs) != 1 or len(unique_rsci_NYs)!= 1:
    print("Images sizes:")
    print("NX:", unique_rsci_NXs)
    print("NY:", unique_rsci_NYs)
    out_error("Incompatible image sizes in SCIENCE list.")
NX = unique_rsci_NXs[0]
NY = unique_rsci_NYs[0]
print("Unique DITS found in science images:", unique_rsci_dits)
print("Unique FILTERS found in science images:", unique_rsci_filters)
print("Found files with a size (%d,%d)"%(NY,NX))
print("There is/are %d targets in list:"%len(unique_rsci_targets), unique_rsci_targets)

# Leer DITS de imagenes flats
if FLAT:
    flat_images = ccdproc.ImageFileCollection(filenames=files_flat, keywords='*')
    flat_dits = []
    flat_filters = []
    for hdu, fname in flat_images.headers(return_fname=True):
        flat_dits.append(hdu['EXPTIME'])
        flat_filters.append(hdu['FILTER'])
    flat_filters = num.char.asarray(flat_filters)
    unique_flat_dits = num.unique(flat_dits)
    unique_flat_filters = num.unique(flat_filters)
    nflats = len(unique_flat_filters)
    print("Unique DITS found in flat images:", unique_flat_dits)
    print("Unique FILTERS found in flat images:", unique_flat_filters)

# Leer DITS de imagenes dark
if DARK:
    dark_images = ccdproc.ImageFileCollection(filenames=files_dark, keywords='*')
    dark_dits = []
    for hdu, fname in dark_images.headers(return_fname=True):
        dark_dits.append(float(hdu['EXPTIME']))
    unique_dark_dits = num.unique(dark_dits)
    ndarks = len(unique_dark_dits)
    print("Unique DITS found in dark images:", unique_dark_dits)

# Verifica que todos los DITS de sciencia poseen un dark para calibrar.
#if DARK:
#    for dit in rsci_dits:
#        if not dit in dark_dits:
#            DARK = False

## Si se solicita bias y dark, se selecciona solo bias
#if BIAS and DARK:
#    DARK = False
#    out_warning("Only BIAS correction will be applied. Changing DARK to FALSE.")

if not (BIAS or DARK):
    out_error("It is not possible to calibrate the science data. Neither BIAS nor DARK available for calibration")

if FLAT:
    for filt in rsci_filters:
        if not filt in flat_filters:
            FLAT = False
            out_error("Filter %s not found in flats"%filt)
    
text = []
if BIAS: text.append("BIAS")
if DARK: text.append("DARK")
if FLAT: text.append("FLAT")
if BPIX: text.append("BPIX")
out_info("Will apply %s"%", ".join(text))

sci_mask = num.zeros((NY,NX), dtype='bool')
sci_mask[scibox[1]-1:scibox[3]-1, scibox[0]-1:scibox[2]-1] = True

if BIAS:
    print("Applying BIAS correction")
    mbias_file = "%s/masterbias.fits"%cal_calibs_dir
    if os.path.exists(mbias_file):  # Read existing file
        print("Reading BIAS from file %s"%mbias_file)
        mbias = CCDData.read(mbias_file)
    else:
        print("Calculating BIAS image")
        mbias = imcombine(files_bias, sigclip=csigclip)
        mbias.write(mbias_file)

if DARK:
    dark = []
    print("Applying DARK correction")
    for dit in unique_dark_dits:
        mdark_file = "%s/masterdark_%.1f.fits"%(cal_calibs_dir,dit)
        if os.path.exists(mdark_file):  # Read existing file
            print("Reading DARK from file %s"%mdark_file)
            dark.append(CCDData.read(mdark_file))
        else:
            print("Calculating DARK image for dit %.1f"%dit)
            idxs = num.asarray(dark_dits) == dit
            files_dark_dit = num.char.asarray(files_dark)[idxs].tolist()
            mdark = imcombine(files_dark_dit, sigclip=csigclip)
            if BIAS:
                mdark = ccdproc.subtract_bias(mdark, mbias)
            mdark.write(mdark_file)
            dark.append(mdark)
                

if FLAT:
    flat = []
    print("Applying FLAT correction")
    mflat_dir  = "%s/flat"%(cal_calibs_dir)
    for filt in unique_flat_filters:
        mflat_file = "%s/masterflat_%s.fits"%(cal_calibs_dir,filt)
        if os.path.exists(mflat_file):  # Read existing file
            print("Reading FLAT from file %s"%mflat_file)
            flat.append(CCDData.read(mflat_file))
        else:
            print("Calculating FLAT image for filter %s"%filt)
            idxs = flat_filters == filt
            files_flat_filter = num.char.asarray(files_flat)[idxs].tolist()
            if BIAS and not DARK:
                mflat = flatcombine(files_flat_filter, bias=mbias, dark_list=None, filt=filt, cdir=mflat_dir, sigclip=csigclip, scienceframe=sci_mask)
            elif BIAS and DARK:
                mflat = flatcombine(files_flat_filter, bias=mbias, dark_list=dark, dark_list_dits=unique_dark_dits, filt=filt, cdir=mflat_dir, sigclip=csigclip, scienceframe=sci_mask, tol=DTOL)
            elif not BIAS and DARK:
                mflat = flatcombine(files_flat_filter, bias=None, dark_list=dark, dark_list_dits=unique_dark_dits, filt=filt, cdir=mflat_dir, sigclip=csigclip, scienceframe=sci_mask, tol=DTOL)
            else:
                out_error("Need either BIAS of DARK")
            mflat.write(mflat_file)
            flat.append(mflat)
            print("Flat done")

"""
if BPIX:
    if DARK:
        if len(dark) > 1:
            idx1 = num.argmin(unique_dark_dits)
            idx2 = num.argmax(unique_dark_dits)
            pixels1 = dark[idx1].data.flatten() / unique_dark_dits[idx1]
            pixels2 = dark[idx2].data.flatten() / unique_dark_dits[idx2]
            plt.figure(figsize=(10, 10))
            plt.plot(pixels1, pixels2, '.', alpha=0.2, label='Data')
            x0 = min(pixels1.min(), pixels2.min())
            x1 = max(pixels1.max(), pixels2.max())
            plt.plot([x0, x1], [x0, x1], label='Ideal relationship')
            plt.xlabel("dark current ($e^-$/sec), %.1f sec exposure time"%unique_dark_dits[idx1])
            plt.ylabel("dark current ($e^-$/sec), %.1f sec exposure time"%unique_dark_dits[idx2])
            plt.grid()
            plt.show()
"""


    
# Apply calibrations to science data
# Iterate through filters
#for filt in unique_flat_filters:
rsci_images.refresh()
csci_filename = []
csci_pathname = []
if not target in unique_rsci_targets:
    out_error("Target %s not in list of SCIENCE images!"%target)
if not os.path.exists(cal_science_dir):
    os.makedirs(cal_science_dir)
for orig_image, fname in rsci_images.ccds(return_fname=True, ccd_kwargs={'unit': 'adu'}):
    target_name = orig_image.header['OBJECT']
    image = ccdproc.trim_image(orig_image[scibox[1]-1:scibox[3]-1, scibox[0]-1:scibox[2]-1])
    if target_name == target:
        if BIAS:
            tbias = ccdproc.trim_image(mbias[scibox[1]-1:scibox[3]-1, scibox[0]-1:scibox[2]-1])
            image_cbias = ccdproc.subtract_bias(image, tbias)
            reduced = image_cbias
        if DARK:
            rsci_dit = float(image.header['EXPTIME'])
            idx = closest_dark_id(unique_dark_dits, rsci_dit, tol=DTOL)
            print("Unique", idx)
            tdark = ccdproc.trim_image(dark[idx][scibox[1]-1:scibox[3]-1, scibox[0]-1:scibox[2]-1])
            if BIAS:
                image_cdark = ccdproc.subtract_dark(image_cbias, tdark, exposure_time="EXPTIME", exposure_unit=u.second)
            else:
                image_cdark = ccdproc.subtract_dark(image, tdark, exposure_time="EXPTIME", exposure_unit=u.second)
            reduced = image_cdark
        if FLAT:
            rsci_filt = image.header['FILTER']
            idx = find_filter_id(unique_flat_filters, rsci_filt)
            tflat = ccdproc.trim_image(flat[idx][scibox[1]-1:scibox[3]-1, scibox[0]-1:scibox[2]-1])
            if BIAS:
                if not DARK:
                    reduced = ccdproc.flat_correct(image_cbias, tflat)
            if DARK:
                reduced = ccdproc.flat_correct(image_cdark, tflat)
        # Trim image
        reduced = ccdproc.trim_image(reduced[scibox[1]-1:scibox[3]-1, scibox[0]-1:scibox[2]-1])
        iname = fname.split('/')[-1]
        if iname.split('.')[-1] == 'fts':
            iname = '.'.join(iname.split('.')[:-1]) + ".fits"
        image_cal_name = "%s/%s"%(cal_science_dir, iname)
        out_info("Writing file %s"%image_cal_name)
        reduced.write(image_cal_name, overwrite=True)
        csci_filename.append(iname)
        csci_pathname.append(image_cal_name)

# Calculate combined images for photometry
#cal_science_dir  = "%s/science/%s/%s"%(dir_cal, target, selec)
images = ccdproc.ImageFileCollection(location=cal_science_dir, filenames=csci_filename, keywords='*')
#combiner = ccdproc.Combiner(images.ccds())
print(images.summary)
median_combfile = "%s/combined_median.fits"%cal_science_dir
mean_combfile = "%s/combined_mean.fits"%cal_science_dir
sum_combfile = "%s/combined_sum.fits"%cal_science_dir
max_combfile = "%s/combined_max.fits"%cal_science_dir

# Median
#combined = combiner.median_combine()
#combined.write(median_combfile, overwrite=True)
#Mean
#combined = combiner.average_combine()
combined = ccdproc.combine(csci_pathname, method='average', ccdkwargs='*', format='fits', mem_limit=8e9)
combined.write(mean_combfile, overwrite=True)
#Sum
#combined = combiner.sum_combine()
#combined.write(sum_combfile, overwrite=True)
# Calculate Max() on-the-fly
combined.data *= 0.0
for image in images.data(ccd_kwargs={'unit': 'adu'}):
    combined.data = num.array([combined.data, image.data]).max(axis=0)
print(combined.data.shape)
combined.write(max_combfile, overwrite=True)


