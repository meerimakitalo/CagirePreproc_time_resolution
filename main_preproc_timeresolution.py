import sys
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import math
import datetime
import gc
import pandas as pd
from tools_preproc import *
from tabulate import tabulate

_, input_file = sys.argv
print(input_file)
parameters = {'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
plt.rcParams.update(parameters)

map_path    = './maps/Julia/'
input_path  = './input/'
output_path = './output/'

output_file = input_file[:-5]+'_PROCESSED'


# Set parameters
Apf2eps     = 7.52
FRN1        = 2
NBFRMIN     = FRN1 + 3
NBLFG       = 4
CR_VARJ     = 120
CR_VMRJ     = 2
CR_THRJ     = 15
frame_per_batch = 10 # Number of frames to process at once. Assuming 60s exposure 45 frames is the max (5 is min)

# Set end of previous acquisition at Nsec before begining of current acquisition
Nsec = 0

# Load maps
PIM_ADU_SAT      = np.ravel(fits.getdata(map_path+'PIM_ADU_SAT.fits'))
PIM_ADU_MAXFIT   = np.ravel(fits.getdata(map_path+'PIM_ADU_MAXFIT.fits'))
PIM_ADU_DYN      = fits.getdata('./maps/dynamique.fits')
PIM_REAL_NONLIN  = np.ravel(fits.getdata(map_path+'PIM_REAL_NONLIN.fits'))
PIM_ADU_BASELINE = PIM_ADU_SAT - PIM_ADU_DYN 


PIM_REAL_SIGFLU = np.nan

PIM_PR_SAT  = fits.open('./maps/carte_persistance.fits')    
PIM_PR_SAT  = np.ravel(PIM_PR_SAT[1].data)
PIM_PR_SATB = np.argwhere(PIM_PR_SAT > 0)


PIM_REAL_PPT1 = fits.getdata(map_path+'PIM_REAL_PT1.fits').T
PIM_REAL_PPT2 = fits.getdata(map_path+'PIM_REAL_PT2.fits').T

PIM_REAL_PPA1 = fits.getdata(map_path+'PIM_REAL_PA1.fits').T
PIM_REAL_PPA2 = fits.getdata(map_path+'PIM_REAL_PA2.fits').T

P_LEVEL       = 1
N_PREC        = 3

# Load pixel maps
indref = fits.getdata('./maps/PixVerts.fits')
indv   = fits.getdata('./maps/PixViolet.fits')

# import du fichier à étudier et transformation en tableau
RAMP_total = fits.open(input_path+input_file)
RAMP_total = RAMP_total[1:] # use only the images froom the fits

# For loop to do the time resolution .fits
# gloabl mask to track saturation between batches
global_saturated_mask = np.zeros(2048 * 2048, dtype=bool)
# sve infromation for the differential frames
previous_last_frame = None

if frame_per_batch > NBFRMIN:
    for i in range(0, np.shape(RAMP_total)[0], frame_per_batch): # frames are processed in batches, if not perfectly divisible last batch will be smaller
        end = min(i + frame_per_batch, np.shape(RAMP_total)[0]) # compute the ending index
        RAMP = RAMP_total[i:end]
        N =  np.shape(RAMP)[0]
        print('Number of frames to process: ',N)

        try:
            T0_RAMP       = datetime.datetime.fromisoformat(RAMP[0].header['HIERARCH ESO DET SEQ UTC']+'00')
            TFIN_PREVRAMP = T0_RAMP - datetime.timedelta(Nsec/3600./24.)
        except:
            TFIN_PREVRAMP = 100.

        Bk = table(RAMP, N)

        gc.collect()

        # Etape 2 : pixels saturés et domaine de calcul du SIGNAL
        POM_FRN_SAT = PixSat(Bk, PIM_ADU_SAT, N, indref)
        FRN_MAXFIT = PlageFit(Bk, PIM_ADU_MAXFIT, N, indref)
        
        # Etape 3: construction de la rampe corrigee
        B3D = tableau3D(RAMP, N)

        t_temps = np.shape(Bk)[0]
        del Bk
        colones_cor = correctionC(B3D)
        del B3D
        lignes_cor = correctionL(colones_cor, NBLFG)
        del colones_cor
        Ck = Tableau2DFlat(lignes_cor, t_temps)
        del lignes_cor
        # Etape 4: construction de la rampe differentielle
        Dk = rampeCDS(Ck, N, prev_frame=previous_last_frame)
        # Calculate the offfset parameter that tracks the passing of time and update the last frame parameter
        if previous_last_frame is None:
            d=2
            offset = i
        else:
            d=1
            offset = FRN1 + i
        previous_last_frame = Ck[-1].copy()
        
        # Etape 5 : pixels touchés par un Rayon cosmique et Etape 6 : estimation du signal en adu/fr
        S_ADU, VAR_ADU, POM_FRN_CR, POM_NBF_FIT, Ac, Bc, Cc, Ncc = FitCosmic(Dk, FRN_MAXFIT, d, PIM_REAL_NONLIN, 
                                                                             PIM_ADU_DYN, POM_FRN_SAT,
                                                                            CR_VARJ, CR_VMRJ, CR_THRJ, NBFRMIN, indv, offset)

        # Etape 7: correction signal
        POM_REAL_SIGNAL = S_ADU
        POM_REAL_VAR = VAR_ADU

        # Update saturation mask 
        current_batch_saturated = (POM_FRN_SAT > 0)
        
        global_saturated_mask = global_saturated_mask | current_batch_saturated

        # Etape 8 : construction de la carte de persistance
        try:
            DT = (T0_RAMP - TFIN_PREVRAMP).seconds
        except:
            DT = 100.
        POM_REAL_PERSIG = CorrectifPersistance(PIM_PR_SATB, N, DT, PIM_REAL_PPA1, PIM_REAL_PPA2, PIM_REAL_PPT1,
                                            PIM_REAL_PPT2, P_LEVEL, N_PREC)

        # Etape 9 : mise a jour des variables utilisées par le preproc
        # Check if a cosmic ray has hit the detector and update the saturation
        new_cr_mask = POM_FRN_CR > 0
        
        POM_FRN_SAT[new_cr_mask] = np.where(
            (POM_FRN_SAT[new_cr_mask] == 0) | (POM_FRN_CR[new_cr_mask] < POM_FRN_SAT[new_cr_mask]),
            POM_FRN_CR[new_cr_mask],
            POM_FRN_SAT[new_cr_mask]
        )

        # Update the presistence
        newly_sat_mask = current_batch_saturated & (~global_saturated_mask)
        newly_sat_indices = np.argwhere(newly_sat_mask)

        if newly_sat_indices.size > 0:
            PIM_PR_SATB = np.vstack((PIM_PR_SATB, newly_sat_indices))
            PIM_PR_SATB = np.unique(PIM_PR_SATB, axis=0)

        PIM_FRN_SAT = POM_FRN_SAT
        
        # Etape 10 : stockage fichier fits
        POM_REAL_SIGNAL[np.isnan(POM_REAL_SIGNAL)] = 0

        POM_FRN_SAT_ABS = np.where(POM_FRN_SAT > 0, POM_FRN_SAT + offset, 0)
        POM_FRN_CR_ABS  = np.where(POM_FRN_CR > 0, POM_FRN_CR + offset, 0)
        FRN_MAXFIT_ABS  = FRN_MAXFIT + offset

        image = np.zeros([7, 2048,2048])
        image[0, :,:] = np.reshape(POM_REAL_SIGNAL, [2048, 2048])
        image[1, :,:] = np.reshape(np.sqrt(POM_REAL_VAR), [2048, 2048])
        image[2, :,:] = np.reshape(POM_FRN_CR_ABS, [2048, 2048])
        image[3, :,:] = np.reshape(POM_FRN_SAT_ABS, [2048, 2048])
        image[4, :,:] = np.reshape(POM_NBF_FIT, [2048, 2048])
        image[5, :,:] = np.reshape(FRN_MAXFIT_ABS, [2048, 2048])
        image[6, :,:] = np.reshape(POM_REAL_PERSIG, [2048, 2048])

        if "plot" in sys.argv:
            plt.figure('Processed image')
            plt.imshow(image[0], vmin=np.quantile(image[0], 0.1), vmax=np.quantile(image[0], 0.9))
            plt.show()
        batch_output = f"{output_path}{output_file}_batch_{i:02d}"
        SaveFit(image, 7, ['Signal', 'VarianceSignal', 'CarteCosmiques', 'PremiereFRSat', 'nbframeFit', 'maxfit', 'PERSIST'],
                batch_output, 'ORIGIN', input_path+input_file)
        del(image)
        gc.collect()

else: # Preprocessing if number of frames is too small
    N = 45
    Bk = table_old(input_path+input_file, N)
    B3D = tableau3D_old(input_path + input_file, N)
    t_temps = np.shape(Bk)[0]
    colones_cor = correctionC(B3D)
    del B3D
    lignes_cor = correctionL(colones_cor, NBLFG)
    del colones_cor
    Ck = Tableau2DFlat(lignes_cor, t_temps)
    del lignes_cor
    Dk = rampeCDS(Ck, N)
    # Initialize Outputs
    S_ADU = np.zeros((2048, 2048))
    VAR_ADU = np.zeros((2048, 2048))
    POM_NBF_FIT = np.zeros((2048, 2048))
    ADU2E = Apf2eps
    for i in range(0, N, frame_per_batch):
        end = min(i + frame_per_batch, N)
        current_batch_size = end - i
        Dk_sub = Dk[i : end]
        Ck_sub = Ck[i : end]
        if frame_per_batch == 1:
            S_ADU = Ck_sub[0] - PIM_ADU_BASELINE
            VAR_ADU = S_ADU / ADU2E
            POM_NBF_FIT[:] = 1

        elif frame_per_batch == 2:
            S_ADU = Dk_sub[1] 
            VAR_ADU = S_ADU / ADU2E
            POM_NBF_FIT[:] = 1

        elif frame_per_batch == 3:
            S_ADU = Dk_sub[2]
            VAR_ADU = S_ADU / ADU2E
            POM_NBF_FIT[:] = 1
            
        elif 4 <= frame_per_batch:
            Dk_subset = Dk_sub[2:] 
            S_ADU = np.median(Dk_subset, axis=0)
            VAR_ADU = (S_ADU / (frame_per_batch - 2)) / ADU2E
            POM_NBF_FIT[:] = 1

        # Prepare other variables to match the structure of the main pipeline, needed to save the .fits
        POM_FRN_CR = np.zeros((2048, 2048))
        POM_FRN_SAT = np.zeros((2048, 2048))
        FRN_MAXFIT = np.full((2048, 2048), N)
        POM_REAL_PERSIG = np.zeros((2048, 2048))
        POM_REAL_SIGNAL = S_ADU
        POM_REAL_SIGNAL[np.isnan(POM_REAL_SIGNAL)] = 0
        POM_REAL_VAR = VAR_ADU

        image = np.zeros([7, 2048,2048])
        image[0, :,:] = np.reshape(POM_REAL_SIGNAL, [2048, 2048])
        image[1, :,:] = np.reshape(np.sqrt(POM_REAL_VAR), [2048, 2048])
        image[2, :,:] = np.reshape(POM_FRN_CR, [2048, 2048])
        image[3, :,:] = np.reshape(POM_FRN_SAT, [2048, 2048])
        image[4, :,:] = np.reshape(POM_NBF_FIT, [2048, 2048])
        image[5, :,:] = np.reshape(FRN_MAXFIT, [2048, 2048])
        image[6, :,:] = np.reshape(POM_REAL_PERSIG, [2048, 2048])

        batch_output = f"{output_path}{output_file}_batch_{i:02d}"
        SaveFit(image, 7, ['Signal', 'VarianceSignal', 'CarteCosmiques', 'PremiereFRSat', 'nbframeFit', 'maxfit', 'PERSIST'],
                batch_output, 'ORIGIN', input_path+input_file)
        del(image)
        gc.collect()