import numpy as num
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.patches as patches
import astropy.io.fits as fits
import astropy.convolution as conv
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.time import Time
from astropy.coordinates import EarthLocation
import glob
import sys
import os
from cc_transits import *
from photutils.detection import DAOStarFinder
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, ApertureStats, EllipticalAperture
from photutils.morphology import data_properties
import barycorrpy

def find_stars(image, photdata, nstars, sr=10.0, nsig=3.0, med=0.0, std=1.0, niter=3):
    out = []
    coords = []
    prob = num.ones(nstars, dtype=bool)
    for i in range(nstars):
        fwhm = 0.0
        airmass = 0.0
        bx0 = int(photdata[i][0])
        bx1 = int(photdata[i][1])
        by0 = int(photdata[i][2])
        by1 = int(photdata[i][3])
        x0 = float(photdata[i][4]) - bx0
        y0 = float(photdata[i][5]) - by0
        img = image[by0:by1,bx0:bx1]
        if not num.all(num.isfinite(img)):
            out_error("Star %d presents unvalid pixels (NaN, Inf)"%i)
        if med == 0.0:
            mean, med, std = sigma_clipped_stats(img, sigma=NSIGMA)
        by,bx = img.shape
        xx,yy = num.meshgrid(num.arange(bx),num.arange(by))
        imask1 = img > med + nsig*std
        if not num.all(num.isfinite(img[imask1])):
            out_warning("Not bright star in field %d"%i)        
        imask2 = num.sqrt((xx-x0)**2 + (yy-y0)**2) <= sr
        if not num.all(num.isfinite(img[imask2])):
            out_warning("Not possible to create centering mask in field %d"%i)
        imask = num.logical_and(imask1,imask2)
        if not num.all(num.isfinite(img[imask])):
            out_error("Not possible to create stellar mask in field %d"%i)
        if len(img[imask]) < 5:
            out_warning("Star %d not found in centering algorithm"%i)
            xc = x0
            yc = y0
            prob[i] = False
        else:
            xc = (xx*img)[imask].sum() / img[imask].sum()
            yc = (yy*img)[imask].sum() / img[imask].sum()
            for j in range(niter):
                imask2 = num.sqrt((xx-xc)**2 + (yy-yc)**2) <= sr
                imask = num.logical_and(imask1,imask2)
                xc = (xx*img)[imask].sum() / img[imask].sum()
                yc = (yy*img)[imask].sum() / img[imask].sum()
            #print("Original (%.2f,%.2f) - New (%.2f,%.2f)"%(x0, y0, xc, yc))
            fwhm = num.sqrt(len(img[imask])/num.pi)
        dist = num.sqrt((x0-xc)**2 + (y0-yc)**2)
        if dist > sr:
            out_warning("Position miscalculated in star %d"%i)
            prob[i] = False
        xc += bx0
        yc += by0
        coords.append([xc, yc])
        out.append([xc, yc, med, std, fwhm])
    return coords, out, prob

if len(sys.argv) != 2:
    out_error("Use: python %s PARFILE.par"%sys.argv[0])
else:
    parfile = sys.argv[1]
    par = read_paramfile(parfile)

target       = par.target
target_ra    = par.tg_ra
target_dec   = par.tg_dec
target_pmra  = par.tg_pmra
target_pmdec = par.tg_pmdec
target_paralax = par.tg_parallax
target_rv      = par.tg_rv

instrument = par.instrument
obsdate    = par.obsdate
selec      = par.selec
dir_imgs = "../imagenes/%s/%s"%(instrument,obsdate)
#dir_cal  = "%s/calibrated"%(dir_imgs)
dir_cal  = "%s/calibrated/%s/%s"%(dir_imgs, target, selec)
dir_phot = "../photometry/%s/%s/%s/%s"%(instrument,obsdate,target,selec)
#cal_science_dir  = "%s/science/%s/%s"%(dir_cal, target, selec)
cal_science_dir  = "%s/science"%(dir_cal)
dir_phot_figs  = "%s/figures"%(dir_phot)

photname = "%s/%s.phot"%(dir_phot,parfile.split('.')[:-1][0])
image_max_file = "%s/combined_max.fits"%cal_science_dir
image_avg_file = "%s/combined_mean.fits"%cal_science_dir

inst = define_instrument(instrument)
instrument = par.instrument
gain = inst.gain
ron = inst.ron
obs_lat = inst.obs_lat
obs_long = inst.obs_long
obs_elev = inst.obs_elev
obs = inst.obs

# Photometry parameters
SRAD      = par.srad         # Search radius around given center
THRESHOLD = par.threshold    # Threshold for centering
FWHM      = par.fwhm         # FWHM
NSIGMA    = par.calsigclip   # Sigma clipping
NITER     = par.niter        # Number of iterations in sigma-clip
PLOT_STEP = par.plotstep     # Frequency of centering plots: 0 (no plots but first), 1 (plot every file), N (every N files)


if not os.path.exists(dir_phot_figs):
    os.makedirs(dir_phot_figs)

if os.path.exists(photname):
    print("Reading photometry file %s"%photname)
    print("Star definitions will be taken from this file")
    photdata = num.loadtxt(photname)
    nstars = len(photdata)
    out_info("Photometry will be applied to %d stars in each image"%nstars)
else:
    out_error("File %s not found"%photname)

ap_list  = par.apertures  # Range of apertures
an_list  = par.annuli     # Range of annuli
dan_list = par.dannuli    # Range of annuli widths

# Asumiendo 1 valor por ahora
annulus = an_list   #[0]
dannulus = dan_list #[0]

sci_list = "%s/science_images.lst"%(cal_science_dir)
print("Reading file %s"%sci_list)
if not os.path.exists(sci_list):
    out_error("File %s does not exist.\nPlease go to directory %s and create a list with the files to apply photometry (and call it science_images.lst)."%(sci_list, cal_science_dir))
fd = open(sci_list, "r")
files_sci = fd.read().splitlines()
fd.close()
files_sci_full = ["%s/%s"%(cal_science_dir,files_sci[i]) for i in range(len(files_sci))]
nfiles = len(files_sci_full)
out_info("Photometry will be applied to %d files"%nfiles)

hdu = fits.open(image_max_file)
max_image = hdu[0].data
hdu.close()
mask = num.ones(max_image.shape, dtype=bool)
xycoords = []
for i in range(nstars):
    bx0 = int(photdata[i][0])
    bx1 = int(photdata[i][1])
    by0 = int(photdata[i][2])
    by1 = int(photdata[i][3])
    mask[by0:by1,bx0:bx1] = False
    xycoords.append([photdata[i][4],photdata[i][5]])

nfx = 4
nfy = int(num.ceil(nstars/nfx))
FX,FY = (nfx*3, nfy*3)
fig = plt.figure(figsize=(FX+1,FY+1))


def draw_stars(figname, image, figure, star_info, coords, ri, an, dan):
    if not os.path.exists(figname):
        nstars = len(star_info)
        figure.clf()
        img_values = num.sort(image.flatten())
        vmin = img_values[int(0.05*len(img_values))]
        vmax = img_values[int(0.95*len(img_values))]
        for i in range(nstars):
            iy = int(num.floor(i/nfx))
            ix = int(num.floor(i - iy*nfx))
            ax = plt.subplot2grid((nfy,nfx),(iy,ix), fig=figure)
            bx0 = int(star_info[i][0])
            bx1 = int(star_info[i][1])
            by0 = int(star_info[i][2])
            by1 = int(star_info[i][3])
            x0 = star_info[i][4] - bx0
            y0 = star_info[i][5] - by0
            xc = coords[i][0] - bx0
            yc = coords[i][1] - by0
            img = image[by0:by1,bx0:bx1]
            ny,nx = img.shape
            ax.imshow(img, cmap='gray',origin='lower', vmin=vmin, vmax=vmax)
            ax.plot([x0], [y0], marker='x', color='red', alpha=0.5)
            ax.plot([xc], [yc], marker='x', color='lime', alpha=0.5)
            for r in ri:
                s_ap = patches.Circle((x0, y0), SRAD, facecolor='none', edgecolor='red', linewidth=0.5, alpha=1.0)
                apert = patches.Circle((xc, yc), r, facecolor='none', edgecolor='orange', linewidth=0.3, alpha=0.8)
                annul = patches.Wedge((xc,yc), an+dan, 0.0, 360.0, width=dan, color='blue', alpha=0.02)
                ax.add_patch(s_ap)
                ax.add_patch(annul)
                ax.add_patch(apert)
                ax.text(5,ny-5, "%d"%i, color='lime', ha='left', va='top')
            figure.savefig(figname, dpi=300)
    else:
        out_warning("File %s exists. Not creating centering figures"%figname)

#nfiles = 1
sigclip = SigmaClip(sigma=3.0, maxiters=10)
phot = []
dphot = []
mjd_tdb = num.zeros(nfiles, dtype="float64")
jd_utc = num.zeros(nfiles, dtype="float64")
prob_total = num.ones(nstars, dtype=bool)
params = []
airmass = []
fits_fwhm = []
for i in range(nfiles):
    image_name = files_sci_full[i]
    print(image_name)
    image = CCDData.read(image_name, mask=mask)
    header = image.header
    obsdate = header['DATE-OBS']
    airmass.append(header['AIRMASS'])
    try:
        fwhm = header['FWHM']
    except:
        fwhm = 0.0
    fits_fwhm.append(fwhm)
    time_utc = Time(obsdate, format='fits', scale='utc', location=(obs_lat*u.deg, obs_long*u.deg))
    jd_utc[i] = time_utc.jd
    mjd_tdb[i] = time_utc.tdb.mjd
    mean, median, std = sigma_clipped_stats(image, sigma=NSIGMA)
    print("Mean=%.2f, median=%.2f, std=%.2f"%(mean, median, std))
    #coords, allpars, prob_star = find_stars(image.data, photdata, nstars, sr=SRAD, nsig=THRESHOLD, med=median, std=std, niter=NITER)
    coords, allpars, prob_star = find_stars(image.data, photdata, nstars, sr=SRAD, nsig=THRESHOLD, med=0.0, std=std, niter=NITER)
    prob_total = num.logical_and(prob_total, prob_star)
    fig_aper_name = "%s/%s.png"%(dir_phot_figs, '.'.join(files_sci[i].split('.')[:-1]))
    if i == 0:
        draw_stars(fig_aper_name, image.data, fig, photdata, coords, ap_list, annulus, dannulus)
    else:
        if PLOT_STEP > 0:
            if i%PLOT_STEP==0:
                draw_stars(fig_aper_name, image.data, fig, photdata, coords, ap_list, annulus, dannulus)
    apertures = [CircularAperture(coords, r=r) for r in ap_list]
    annulus_aperture = CircularAnnulus(coords, r_in=annulus, r_out=annulus+dannulus)
    #phot_table = aperture_photometry(image.data, apertures, mask)
    #phot_bkgsub = phot_table['aperture_sum'] - total_bkg
    bkg_stats = ApertureStats(image.data, annulus_aperture, sigma_clip=sigclip)
    aper_flux = []
    aper_dflux = []
    for aperture in apertures:
        aper_stats = ApertureStats(image.data, aperture, sigma_clip=None)
        total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
        apersum_bkgsub = aper_stats.sum - total_bkg
        apersum_error = num.sqrt(apersum_bkgsub/gain + aper_stats.sum_aper_area.value*bkg_stats.std**2 + aper_stats.sum_aper_area.value**2*bkg_stats.std**2/bkg_stats.sum_aper_area.value)
        aper_flux.append(apersum_bkgsub)
        aper_dflux.append(apersum_error)
    phot.append(aper_flux)
    dphot.append(aper_dflux)
    params.append(allpars)

    # for col in phot_table.colnames:
    #     phot_table[col].info.format = '%.8g'  # for consistent table output
    # print(phot_table)
    # daofind = DAOStarFinder(fwhm=FWHM, threshold=THRESHOLD*std, xycoords=xycoords)
    # sources = daofind(image - median)
    # for col in sources.colnames:  
    #     if col not in ('id', 'npix'):
    #         sources[col].info.format = '%.2f'  # for consistent table output
    # sources.pprint(max_width=176)
    # image.wcs = None
    # image.data[~mask] *= 0.0
    # image.write("TEST.fits", overwrite=True)
    
plt.close(fig)

bjd_tdb = barycorrpy.utc_tdb.JDUTC_to_BJDTDB(jd_utc, ra=target_ra, dec=target_dec, pmra=target_pmra, pmdec=target_pmdec, rv=target_rv, obsname=obs, leap_update=True)
bjd_tdb0 = num.floor(bjd_tdb[0][0])
bjd_tdb = bjd_tdb[0] - bjd_tdb0
mjd_tdb0 = num.floor(mjd_tdb[0])
mjd_tdb = mjd_tdb - mjd_tdb0

phot = num.asarray(phot)
dphot = num.asarray(dphot)
airmass = num.asarray(airmass)
fits_fwhm = num.asarray(fits_fwhm)

x = num.arange(len(phot))
napert = len(ap_list)
fig = plt.figure(figsize=(10,8))
stats = []
for ap in range(napert):
    if ap == 0:
        ax = plt.subplot2grid((napert,1),(ap,0), fig=fig)
    else:
        ax = plt.subplot2grid((napert,1),(ap,0), fig=fig, sharex=ax)
    for i in num.arange(nstars-1)+1:
        f0 = phot[:,ap,0]
        f1 = phot[:,ap,i]
        df0 = dphot[:,ap,0]
        df1 = dphot[:,ap,i]
        f = f0/f1
        df = num.sqrt((df0/f1)**2 + (f0/f1**2*df1)**2)
        std = num.std(f)
        mean = num.mean(f)
        f /= mean
        #f += 3*std*(i-1)
        if prob_total[i]:
            lstyle = '-'
        else:
            lstyle = '--'
        ax.plot(bjd_tdb, f, linestyle=lstyle)
        ax.errorbar(bjd_tdb, f, yerr=df, capsize=4, fmt='o', ms=4)
        ax.tick_params(bottom=True, labelbottom=False, direction='inout')
        stats.append([ap_list[ap], i, num.std(f)])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    x_txt = 0.97 * (x1-x0) + x0
    y_txt = 0.80 * (y1-y0) + y0        
    ax.text(x_txt,y_txt, "Ap. %.1f"%ap_list[ap], color='black', ha='center', va='center', fontsize='8')
ax.set_xlabel("BJD(TDB) - %.0f (d)"%bjd_tdb0)
ax.tick_params(bottom=True, labelbottom=True)
fig.subplots_adjust(hspace=0, left=0.1, right=0.99, bottom=0.1, top=0.99)
fig_all_lightcurves = "%s/all_lightcurves.png"%(dir_phot)
plt.savefig(fig_all_lightcurves, dpi=300)
plt.show()

for ap in range(napert):
    phot_file1 = "%s/photometry_individual_ap%.1f.dat"%(dir_phot, ap_list[ap])
    phot_file2 = "%s/lightcurves_ap%.1f.dat"%(dir_phot, ap_list[ap])
    f0 = phot[:,ap,0]
    df0 = dphot[:,ap,0]    
    phot_data1 = num.column_stack([num.ones(nfiles)*bjd_tdb0, bjd_tdb, f0, df0])
    phot_data2 = num.column_stack([num.ones(nfiles)*bjd_tdb0, bjd_tdb])
    text_head1 = "%6s %12s %15s %15s"%("BJD0", "BJD", "flux0", "dflux0")
    text_head2 = "%6s %12s"%("BJD0", "BJD")
    text_fmt1 = "%8.0f %12.10f %15.8f %15.8f"
    text_fmt2 = "%8.0f %12.10f"
    for i in num.arange(nstars-1)+1:
        fi = phot[:,ap,i]
        dfi = dphot[:,ap,i]
        df0i = num.sqrt((df0/fi)**2 + (f0/fi**2*dfi)**2)
        phot_data1 = num.insert(phot_data1, 2+2*i, fi, axis=1)
        phot_data1 = num.insert(phot_data1, 2+2*i+1, dfi, axis=1)
        phot_data2 = num.insert(phot_data2, 1+2*i-1, f0/fi, axis=1)
        phot_data2 = num.insert(phot_data2, 1+2*i, df0i, axis=1)
        text_head1 += " %15s %15s"%("flux%d"%i, "dflux%d"%i)
        text_head2 += " %12s %12s"%("rflux%d"%i, "drflux%d"%i)
        text_fmt1 += " %15.8f %15.8f"
        text_fmt2 += " %12.8f %12.8f"
    num.savetxt(phot_file1, phot_data1, fmt=text_fmt1, header=text_head1)
    num.savetxt(phot_file2, phot_data2, fmt=text_fmt2, header=text_head2)

text_head = "%8s %10s %10s %10s %10s %10s %10s"%("xcen", "ycen", "skymed", "skystd", "fwhm", "airmass", "fwhm_hdr")
text_fmt  = "%10.5f %10.5f %10.4f %10.4f %10.6f %10.6f %10.6f"
for i in range(nstars):
    param_file = "%s/parameters_star%d.dat"%(dir_phot,i)
    param_data = []
    for j in range(nfiles):
        param_data.append(params[j][i])
    param_data = num.insert(param_data, 5, airmass, axis=1)
    param_data = num.insert(param_data, 6, fits_fwhm, axis=1)
    num.savetxt(param_file, param_data, fmt=text_fmt, header=text_head)

    
stats = num.asarray(stats)
idx = num.argsort(stats[:,2])
initial_stdout = sys.stdout
log_file = "%s/%s.log"%(dir_phot,parfile.split('.')[:-1][0])
foutput = open(log_file, 'w')
sys.stdout = foutput
print("Sorted photometry:")
for i in range(len(stats)):
    print("Aperture: %.1f, Reference %d - std = %.5f"%(stats[idx][i][0], stats[idx][i][1], stats[idx][i][2]))
foutput.close()
sys.stdout = initial_stdout

out_info("Photometry done!")
out_info("Check final results in %s"%log_file)
