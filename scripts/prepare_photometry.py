import numpy as num
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.patches as patches
import astropy.io.fits as fits
import astropy.convolution as conv
from astropy.io import ascii
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import sys
import os
import shutil

from cc_transits import *


global BBOX
global BOX0
global NSIG_FIND
global LIMCLIP
global KER_SIG
global RAD_FIND


#############
BBOX = 50       # Final box size
BOX0 = 20        # Search box size
NSIG_FIND = 10   # N*STDV to find stars
LIMCLIP  = 0.1    # % of data to clip in sky calculation
AP_RAD   = 10       # Aperture radius to show
KER_SIG  = 0      # Width of search kernel (do not change here)
RAD_FIND = 10    # Radius to calculate centers (once a star is found)
NIter    = 3        # Number of times search is repeated per position
#############

# To pick stars in images
class FoV_mask:
    def __init__(self, figure, ax, image):
        self.figure = figure
        self.stars_x = []
        self.stars_y = []
        self.stars_info = []
        self.key_id = figure.canvas.mpl_connect('key_press_event', self)
        self.mouse_in  = figure.canvas.mpl_connect('axes_enter_event', self)
        self.ax = ax
        self.image = image
        self.i_star = 0
        self.available = False
        self.xc = 0.0
        self.yc = 0.0
        self.star_info = []
        self.fbox = []
        self.atpos = False
    def __call__(self, event):
        if (event.inaxes == self.ax):
            if event.key in ['p', 'z']:  # Find star around location
                print("Picking star at (%6.1f,%6.1f) pix"%(event.xdata, event.ydata))
                if event.key == 'z':
                    self.atpos = True
                    print("Set atpos to True")
                if event.key == 'p':
                    self.atpos = False
                    print("Set atpos to False")
                self.star_info = determine_box(self.image, event.xdata, event.ydata, self.atpos)
                self.xc = self.star_info[4]
                self.yc = self.star_info[5]
                if len(self.fbox) > 0:
                    self.fbox[0][0].remove()
                    self.fbox[1].remove()
                    self.fbox[2].remove()
                if self.xc > 1.0 and self.yc > 1.0 and event.xdata > 1.0 and event.ydata > 1.0:
                    self.fbox = draw_box(self.figure, self.ax, self.i_star, self.star_info)
                    if self.i_star == 0:
                        self.available = True
                    else:
                        for i in range(len(self.stars_info)):
                            if (num.sqrt((self.xc - self.stars_info[i][5])**2 + (self.yc - self.stars_info[i][6])**2)) < AP_RAD:
                                print("Star already picked")
                                self.available = False
                                self.fbox[0][0].remove()
                                self.fbox[1].remove()
                                self.fbox[2].remove()
                        else:
                            self.available = True
                else:
                    self.available = False
                    out_info("Not a valid star at (%.1f,%.1f)"%(event.xdata, event.ydata))
 
            elif event.key in ['a'] and self.available:
                out_info("Adding star %d at (%.1f,%.1f)"%(self.i_star, self.xc, self.yc))
                self.stars_info.append(self.star_info)
                self.i_star += 1
                self.available = False
                self.fbox = []

    def get_positions(self):
        self.figure.canvas.mpl_disconnect(self.mouse_in)
        self.figure.canvas.mpl_disconnect(self.key_id)
        return self.stars_info

def draw_box(figure, ax, i_star, star_info):
    bbox = star_info[0:4]
    xc = star_info[4]
    yc = star_info[5]
    apert = patches.Circle((xc, yc), AP_RAD, facecolor='none', edgecolor='lime')
    rectx = [bbox[0], bbox[1],bbox[1],bbox[0],bbox[0]]
    recty = [bbox[2], bbox[2],bbox[3],bbox[3],bbox[2]]
    box = ax.plot(rectx, recty, linestyle='-', color='lime')
    circ = ax.add_patch(apert)
    label = ax.text(bbox[0]+5,bbox[3]-5, "%d"%i_star, color='lime', ha='left', va='top')
    figure.canvas.draw()
    return [box, circ, label]

    
class FoV_index:
    def __init__(self, photname, starmask, figure):
        self.photname = photname
        self.starmask = starmask
        self.figure = figure
    def phot_save(self, event):
        out_info("Saving photometry data to file %s"%self.photname)
        phot_data = starmask.get_positions()
        num.savetxt(photname, phot_data, fmt="%6d %6d %6d %6d %10.3f %10.3f %10.2f %10.3f %10.3f %10.3f %10.3f")
        figname = "/".join(photname.split('/')[:-1])+"/"+photname.split('/')[-1].split('.')[0]+"_field.png"
        plt.get_current_fig_manager().toolbar.home()
        self.figure.canvas.draw()
        self.figure.savefig(figname, dpi=300)
        plt.close(self.figure)
    def exit(self, event):
        plt.close(self.figure)

def determine_box(image,x0,y0, atpos=False):
    ny,nx = image.shape
    # Box inicial
    bx0 = max(int(x0)-BOX0, 0)
    bx1 = min(int(x0)+BOX0, nx-1)
    by0 = max(int(y0)-BOX0, 0)
    by1 = min(int(y0)+BOX0, ny-1)
    bimg = image[by0:by1,bx0:bx1]
    by,bx = bimg.shape
    x = num.arange(bx)
    y = num.arange(by)    
    xx,yy = num.meshgrid(x,y)
    nb = by*bx
    bimg_flat = num.sort(bimg.flatten())
    mean, sky, stdv = sigma_clipped_stats(bimg, sigma=3.0)
    #sky = num.median(bimg_flat[int(LIMCLIP*nb):int((1-LIMCLIP)*nb)])
    #stdv = num.std(bimg_flat[int(LIMCLIP*nb):int((1-LIMCLIP)*nb)])
    pos = num.unravel_index(num.argmax(bimg, axis=None), bimg.shape)
    if not atpos:
        mask = num.sqrt((xx-pos[1])**2 + (yy-pos[0])**2) < RAD_FIND
        xc = (xx*bimg)[mask].sum() / bimg[mask].sum() + bx0
        yc = (yy*bimg)[mask].sum() / bimg[mask].sum() + by0
        if bimg[pos] > sky+NSIG_FIND*stdv:
            print("It is a star")
            for i in range(NIter):
                bx0 = max(int(xc)-BOX0, 0)
                bx1 = min(int(xc)+BOX0, nx-1)
                by0 = max(int(yc)-BOX0, 0)
                by1 = min(int(yc)+BOX0, ny-1)
                bimg = image[by0:by1,bx0:bx1]
                by,bx = bimg.shape
                x = num.arange(bx)
                y = num.arange(by)    
                xx,yy = num.meshgrid(x,y)
                nb = by*bx
                bimg_flat = num.sort(bimg.flatten())
                mean, sky, stdv = sigma_clipped_stats(bimg, sigma=3.0)
                #sky = num.median(bimg_flat[int(LIMCLIP*nb):int((1-LIMCLIP)*nb)])
                #stdv = num.std(bimg_flat[int(LIMCLIP*nb):int((1-LIMCLIP)*nb)])
                pos = num.unravel_index(num.argmax(bimg, axis=None), bimg.shape)
                bbox = [bx0,bx1,by0,by1]
                mask = num.sqrt((xx-pos[1])**2 + (yy-pos[0])**2) < RAD_FIND
                xc = (xx*bimg)[mask].sum() / bimg[mask].sum() + bx0
                yc = (yy*bimg)[mask].sum() / bimg[mask].sum() + by0
                ###
            pvalue = bimg[pos]
            bx0 = max(int(xc)-BBOX, 0)
            bx1 = min(int(xc)+BBOX, nx-1)
            by0 = max(int(yc)-BBOX, 0)
            by1 = min(int(yc)+BBOX, ny-1)
            bbox = [bx0,bx1,by0,by1]
            xpv = pos[1] + bx0
            ypv = pos[0] + by0
            return [bbox[0],bbox[1],bbox[2],bbox[3], xc,yc,pvalue,xpv,ypv,sky,stdv]
        else:
            print("No stars in pointing")
            return 0,0,0,0, -1, -1, 0, 0, 0, 0, 0
    else:
        xc = x0
        yc = y0
        pvalue = bimg[pos]
        bx0 = max(int(xc)-BBOX, 0)
        bx1 = min(int(xc)+BBOX, nx)
        by0 = max(int(yc)-BBOX, 0)
        by1 = min(int(yc)+BBOX, ny)
        bbox = [bx0,bx1,by0,by1]
        xpv = pos[1] + bx0
        ypv = pos[0] + by0
        return [bbox[0],bbox[1],bbox[2],bbox[3], xc,yc,pvalue,xpv,ypv,sky,stdv]

    
if len(sys.argv) != 2:
    out_error("Use: python %s PARFILE.par"%sys.argv[0])
else:
    parfile = sys.argv[1]
    par = read_paramfile(parfile)

target     = par.target
instrument = par.instrument
obsdate    = par.obsdate
selec      = par.selec
dir_imgs = "../imagenes/%s/%s"%(instrument,obsdate)
#dir_cal  = "%s/calibrated"%(dir_imgs)
dir_cal  = "%s/calibrated/%s/%s"%(dir_imgs, target, selec)
dir_phot = "../photometry/%s/%s/%s/%s"%(instrument,obsdate,target,selec)
#cal_science_dir  = "%s/science/%s/%s"%(dir_cal, target, selec)
cal_science_dir  = "%s/science"%(dir_cal)

if not os.path.exists(dir_phot):
    out_info("Creating directory %s"%dir_phot)
    os.makedirs(dir_phot)

photname = "%s/%s.phot"%(dir_phot,parfile.split('.')[:-1][0])
image_max_file = "%s/combined_max.fits"%cal_science_dir
image_avg_file = "%s/combined_mean.fits"%cal_science_dir

shutil.copy2(parfile, dir_phot)

hdu_list1 = fits.open(image_max_file)
hdu_list2 = fits.open(image_avg_file)
image_max = hdu_list1[0].data
image_avg = hdu_list2[0].data
hdu_list1.close()
hdu_list2.close()
img_idx = 0
images = [image_max, image_avg]
image_actives = [True, False]

fig = plt.figure(figsize=(11,8))
ax = fig.add_axes([0.15, 0.1, 0.85, 0.85])
btn2 = fig.add_axes([0.02,0.87,0.13,0.05])
chk_img = fig.add_axes([0.02, 0.30, 0.17, 0.10])
starmask = FoV_mask(fig,ax,images[img_idx])
callback = FoV_index(photname,starmask,fig)
button2 = widgets.Button(btn2, "Exit without saving", color='sandybrown')
button2.on_clicked(callback.exit)
img_values = num.sort(images[img_idx].flatten())
vmin = img_values[int(0.05*len(img_values))]
vmax = img_values[int(0.95*len(img_values))]    
cax = ax.imshow(images[img_idx], cmap='gray',origin='lower', vmin=vmin, vmax=vmax)

def submit1(val):  # Threshold button
    global NSIG_FIND
    NSIG_FIND = float(val)
    if NSIG_FIND < 1.0:
        out_info("Parameter KER_FIND needs to be > 1. Assuming KER_FIND = 3.0")
        NSIG_FIND = 3.0
    else:
        out_info("Changed NSIG_FIND to %.1f"%NSIG_FIND)

def submit2(val):   # Sigma of convolution kernel button
    global KER_SIG
    global image_conv
    KER_SIG = float(val)
    if KER_SIG > 0:
        kernel = conv.Gaussian2DKernel(x_stddev=KER_SIG)
        image_conv = conv.convolve_fft(images[img_idx], kernel)
        cax.set_data(image_conv)
        fig.canvas.flush_events()
        out_info("Changed KER_SIG to %.1f"%(KER_SIG))
    elif KER_SIG == 0:
        out_warning("Set KER_SIG to 0")
    else:
        out_warning("Invalid value for KER_SIG. Assuming KER_SIG = 0")
        KER_SIG = 0

def submit3(val):   # Search box button
    global BOX0
    BOX0 = int(val)
    if BOX0 < 20:
        out_info("Invalid value for BOX0. Assuming BOX0 = 20")
        BOX0 = 20
    else:
        out_info("Changed BOX0 to %d"%BOX0)

def submit4(val):
    global BBOX
    BBOX = int(val)
    if BBOX < 20:
        out_info("Invalid value for BBOX. Assuming BBOX = 20")
        BBOX = 20
    else:
        out_info("Changed BBOX to %d"%BBOX)

def submit5(val):
    global AP_RAD
    AP_RAD = int(val)
    if AP_RAD < 3:
        out_info("Parameter AP_RAD has to be > 3 pixels. Assuming AP_RAD = 3")
        AP_RAD = 3
    else:
        out_info("Changed AP_RAD to %d"%AP_RAD)

def submit6(val):
    global RAD_FIND
    RAD_FIND = int(val)
    if RAD_FIND < 3:
        out_info("Parameter RAD_FIND has to be > 3 pixels. Assuming RAD_FIND = 3")
        RAD_FIND = 3
    else:
        out_info("Changed RAD_FIND to %d"%RAD_FIND)

check_labels =["Use mean instead\nof max?"]
check_image = widgets.CheckButtons(ax=chk_img, labels=check_labels, actives=[False])
def call_check_image(sel):
    global image_actives
    global img_idx
    status = check_image.get_status()[0]
    if status:
        img_idx = 1
        image_actives = [False, True]
        out_info("Changing image from max to mean")
    else:
        img_idx = 0
        image_actives = [True,False]
        out_info("Changing image from mean to max")
    img_values = num.sort(images[img_idx].flatten())
    vmin = img_values[int(0.05*len(img_values))]
    vmax = img_values[int(0.95*len(img_values))]    
    cax.set_data(images[img_idx])
    cax.set(clim=(vmin,vmax))
    #fig.canvas.flush_events()
    fig.canvas.draw()
check_image.on_clicked(call_check_image)

out_info("Looking for file photometry file %s"%photname)

# Step 1: prepare windows for photometry
# Step 2: calculate parameters for photometry: center, peak value, position of peak, fwhm, sky median, sky std.
if os.path.exists(photname):
    print("Reading photometry file %s"%photname)
    print("Star definitions will be taken from this file")
    photdata = num.loadtxt(photname)
    for i in range(len(photdata)):
        draw_box(fig, ax, i, photdata[i])
    plt.show()
else:
    out_info("File %s not found"%photname)
    out_info("Setting selection of coordinates")
    btn1 = fig.add_axes([0.02,0.93,0.13,0.05])
    txt1 = fig.add_axes([0.10,0.78,0.06,0.05])
    txt2 = fig.add_axes([0.10,0.72,0.06,0.05])
    txt3 = fig.add_axes([0.10,0.66,0.06,0.05])
    txt4 = fig.add_axes([0.10,0.60,0.06,0.05])
    txt5 = fig.add_axes([0.10,0.54,0.06,0.05])
    txt6 = fig.add_axes([0.10,0.48,0.06,0.05])
    button1 = widgets.Button(btn1, "Save and exit", color='lime')
    button1.on_clicked(callback.phot_save)
    text_box1 = widgets.TextBox(txt1,"Threshold", initial=NSIG_FIND, textalignment='left', label_pad=0.1)
    text_box1.on_submit(submit1)
    text_box2 = widgets.TextBox(txt2,"Kernel Sig.", initial=KER_SIG, textalignment='left', label_pad=0.1)
    text_box2.on_submit(submit2)
    text_box3 = widgets.TextBox(txt3,"Search Box", initial=BOX0, textalignment='left', label_pad=0.1)
    text_box3.on_submit(submit3)
    text_box4 = widgets.TextBox(txt4,"Bounding Box", initial=BBOX, textalignment='left', label_pad=0.1)
    text_box4.on_submit(submit4)
    text_box5 = widgets.TextBox(txt5,"Aperture", initial=AP_RAD, textalignment='left', label_pad=0.1)
    text_box5.on_submit(submit5)
    text_box6 = widgets.TextBox(txt6,"Find. Radius", initial=RAD_FIND, textalignment='left', label_pad=0.1)
    text_box6.on_submit(submit6)
    plt.figtext(0.02, 0.20, "Press 'p' to search a star", fontsize=12, figure=fig, color='red')
    plt.figtext(0.02, 0.17, "Press 'z' to fix a star", fontsize=12, figure=fig, color='red')
    plt.figtext(0.02, 0.115, "Press 'a' to accept position\nof picked star", fontsize=12, figure=fig, color='red')
    plt.figtext(0.02, 0.085, "To zoom use the toolbar", fontsize=12, figure=fig, color='red')
    plt.figtext(0.02, 0.055, "Press 'r' to restore zoom", fontsize=12, figure=fig, color='red')
    plt.show()
    photdata = starmask.get_positions()
       
# Step 3: apply photometry on target + N references
nstars = len(photdata)
print("%d stars to be analyzed."%nstars)
