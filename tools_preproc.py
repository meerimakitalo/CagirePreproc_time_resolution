import sys
from astropy.io import fits
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import math


def table_old(fichier, fin):
    fit = fits.open(fichier)
    tab = np.zeros([fin, 4194304])
    for i in range(1, fin + 1):
        tab[i - 1] = np.ravel(fit[i].data)
    return (tab)
def table(ramp_block, N): # table function modified to use ramps not the file
    tab = np.zeros([N, 4194304])
    for i in range(N):
        tab[i] = np.ravel(ramp_block[i].data)
    return tab

def table_simu(fichier, fin):
    fit = fits.open(fichier)
    tab = np.zeros([fin, 4194304])
    f = fit[0].data
    for i in range(0, fin):
        tab[i] = np.ravel(f[i, :, :])
    return (tab)


def PixSat(image, sat, N, indref):
    """
    PixSat = np.argwhere(image[fin - 1, :] >= sat)
    FrameSat = np.sum(np.where(image < sat, 1, 0),axis=0)
    carteP = np.zeros(2048 * 2048)
    carteP[PixSat] = FrameSat[PixSat]+1
    carteP[indref] = 1
    """
    PixSat = np.argwhere(image[N - 1, :] > sat)
    prev = np.array([True for i in range(2048*2048)])
    indMax = np.asarray([0 for i in range(2048*2048)])
    for i in range(N):
        img = np.int64(np.ravel(image[i]))
        indSat = np.where((img > sat) & prev)[0]
        indMax[indSat] = i+1
        prev[indSat] = False
    indMax[indref] = 1
    return indMax


def PlageFit(image, sat70, N, indref):
    prev = np.array([True for i in range(2048*2048)])
    indMax = np.asarray([N for i in range(2048*2048)])
    for i in range(N):
        img = np.ravel(image[i])
        indSat = np.where((img > sat70) & prev)[0]
        indMax[indSat] = i
        prev[indSat] = False
    return indMax


def tableau3D_old(fichier, fin):
    fit = fits.open(fichier)
    tab = np.zeros([fin, 2048, 2048])
    for i in range(1, fin + 1):
        a = fit[i].data
        tab[i - 1] = (a).astype(np.int64)
    return (tab)
def tableau3D(ramp_chunk, N): # tableau3D function modified to use ramps not the file
    tab = np.zeros([N, 2048, 2048], dtype=np.int64)
    for i in range(N):
        tab[i] = ramp_chunk[i].data.astype(np.int64) 
    return tab


def tableau3D_simu(fichier, fin):
    fit = fits.open(fichier)
    f = fit[0].data
    tab = np.zeros([fin, 2048, 2048])
    for i in range(0, fin):
        a = f[i, :, :]
        tab[i] = (a).astype(np.int64)
    return (tab)


def correctionC(image):
    # creation d'un masque  pour selectionner les colonnes des pixels de référence
    l = [1, 2, 2045, 2046]

    # on découpe l'image en 32 voies et on calcul la médiane à appliquer sur chaque voie avec le masque
    split = np.split(image[:, l, :], 32, axis=2)  # *masque

    Corr = np.nanmean(split, axis=(2, 3))

    del split
    Corr = np.transpose(Corr)

    # création d'une cartographie des corrections
    C = np.repeat(Corr, 64, axis=1)
    axe = np.zeros(2048)
    Corr = (axe[:, np.newaxis] + C[:, np.newaxis, :])
    del C
    del axe

    # image corrigée sur les colonnes
    Colonnes_Cor = image - Corr
    del Corr
    del image

    return (Colonnes_Cor)


def correctionL(image, NBLFG):
    # création du masque de selection des pixels de référence
    c = [0, 1, 2, 2045, 2046, 2047]

    # ATTENTION, CETTE FONCTION N'EST ICI PAS ADAPTÉE SI ON CHANGE NBLFG : A AJUSTER

    image[:, (20 - NBLFG):(20 + NBLFG + 1), c] = 0


    # création d'un masque et calcul des moyennes des 3 lignes au dessus et au dessous de la ligne considérée
    CorrL = [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]

    for k in range(NBLFG + 1, 2047 - NBLFG):
        # on calcul la médiane des pix de ref sur chauqe ligne
        CorrL += [(np.nanmean(image[:, (k - NBLFG):(k + NBLFG + 1), c], axis=(1, 2)))]

    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL = np.array(CorrL)


    # création d'une cartographie des correction
    CorrL = CorrL[:, :, np.newaxis]
    Corr = np.repeat(CorrL, 2048, axis=2)
    del CorrL
    CorrL = np.moveaxis(Corr, 1, 0)
    CorrL[np.isnan(CorrL)] = 0

    lignes_cor = image - CorrL

    del CorrL

    return (lignes_cor)


def Tableau2DFlat(image, fin):
    tab = np.zeros([fin, 4194304])
    for i in range(0, fin):
        a = image[i, :, :]
        tab[i] = np.ravel(a).astype(np.float32)
    return (tab)


def rampeCDS_old(image, fin):
    imCDS = np.zeros((fin, 2048 * 2048))
    # on met en première position l'image n°0 corrigée des pixels de ref
    imCDS[0] = image[0]  # on met en première position l'image n°0 corrigée du bias et des pixels de ref
    for i in range(0, fin - 1, 1):
        # on crée la rampe CDS corrigée des pixels de ref
        imCDS[i + 1] = ((image[i + 1]) - (image[i]))
    return (imCDS)
def rampeCDS(image, fin, prev_frame=None): # rampeCDS modified to take into account the previos last frame --> the batches do not always start from frame number 0
    imCDS = np.zeros((fin, 2048 * 2048))
    if prev_frame is not None:
        imCDS[0] = image[0] - prev_frame
    else:
        imCDS[0] = image[0]  
    for i in range(0, fin - 1, 1):
        # on crée la rampe CDS corrigée des pixels de ref
        imCDS[i + 1] = ((image[i + 1]) - (image[i]))
    return (imCDS)

# FitCosmic function modified to take into account every batch does not start from the frame 0
def FitCosmic(image, indSat, d, alpha, sat, framesat, CR_VARJ, CR_VMRJ, CR_THRJ, NBFRMIN, indv, offset=0):
    PixaTraiter = indv
    cos = []
    im = image.copy()
    madLim = np.zeros(2048 * 2048)
    Med = np.zeros(2048 * 2048)
    vr = np.zeros(2048 * 2048)
    fr = np.zeros(2048 * 2048)
    F = np.zeros([1, 4194304])  # matrice des résultats du fit
    varO = np.zeros(4194304)  # variance sur l'offset
    carteCos = np.zeros(2048 * 2048)
    POM_NBF_FIT = np.zeros(2048 * 2048)

    # on prend ici un coef de non linéarité à partir d'un fichier de carac  / ou le fichier test pour ratir
    
    Ac, Bc, Cc, Ncc = [np.zeros(2048 * 2048) for i in range(4)]
    
    for i in np.unique(indSat):
        if i - NBFRMIN >= 0:
            pix_isat = np.intersect1d(np.argwhere(indSat == i), PixaTraiter)
            pix_isat = pix_isat[:, np.newaxis]
            longueur = (np.arange(d-1, i)).astype(np.int64)
            y1 = im[longueur, pix_isat]

            # selection des pixels non eratics et var/flux ok
            v = np.var(y1, axis=1)
            vr[pix_isat[:, 0]] = v

            f = np.mean(y1, axis=1)
            fr[pix_isat[:, 0]] = f

            CR = np.argwhere(v > CR_VARJ)  # 300 #calibration : 120
            g = np.argwhere(v / f > CR_VMRJ)  # 7 #calibration : 2
            indCR = np.intersect1d(g, CR)
            pix = pix_isat[indCR]
            y = im[longueur, pix]

            # calcul de la limite à partir de la mediane et de la MAD
            med = np.median(y, axis=1)
            med2 = med[:, np.newaxis]
            med2 = np.repeat(med2, np.shape(y)[1], axis=1)
            mad = np.median(np.absolute(y - med2), axis=1)
            madLim[pix[:, 0]] = med + CR_THRJ * mad
            Med[pix[:, 0]] = med

            m = madLim[pix[:, 0]]
            m = m[:, np.newaxis]
            m = np.repeat(m, np.shape(y)[1], axis=1)
            
            cr = np.argwhere(y > m)
            p = cr[:, 0]  # pixels impactés
            t = cr[:, 1]  # temps de l'impact
            

            # selection des cosmics : un seul point au dessus de la limite
            uniq, u = np.unique(p, return_counts=True)
            val = uniq[np.argwhere(u == 1)]
            uni = np.argwhere(p == val)

            if np.shape(uni)[0] != 0:
                un = uni[:, 1]
                cr = cr[un]
                tps = cr[:, 1] + d - 1
                p = cr[:, 0]
                pixel = pix[p]
                # on met le temps d'impact dans la carte des pixels impactés
                carteCos[pixel[:, 0]] = tps
                # on supprime le pixel impacté de la liste des pixels normaux
                pix_isat = np.setdiff1d(pix_isat, pixel)[:, np.newaxis]

                # on parcours les cosmics selectionés
                a0 = np.zeros(len(tps))
                er = np.zeros(len(tps))
                Nbfit = np.zeros(len(tps))
                for c in range(0, len(tps)):
                    med = np.nanmedian(im[:, pixel[c, 0]])
                    if med == 0: med = 1
                    l = round(im[tps[c], pixel[c, 0]] / med)
                    FRN_MAXFIT = i + offset
                    FRN1       = d + offset
                    tps_abs    = tps[c] + offset
                    #FRN_MAXFIT = i
                    #FRN1       = d
                    yf = im[longueur, pixel[c, 0]]
                    Al = alpha[pixel[c, 0]]
                    Nc = FRN_MAXFIT - FRN1 + 1
                    A = (Al / 2) * (FRN_MAXFIT**2 - FRN_MAXFIT + 2.*FRN_MAXFIT*l -2.*(tps_abs+1-FRN1+1)*l - FRN1**2 - FRN1)
                    B = Nc - 1
                    C = - np.sum(yf) + im[tps[c], pixel[c, 0]]
                    delta = B * B - 4 * A * C
                    
                    a0[c] = (-B + np.sqrt(delta)) / (2 * A)

                    A1 = Al * (a0[c] ** 2)
                    A1 = np.repeat(A1, len(longueur))
                    Ek = yf - A1 * (longueur+1+offset)
                    er[c] = np.var(np.delete(Ek,tps[c]+1-(d))) / (Nc)
                    Nbfit[c] = Nc - 1
                    
                    Ac[pixel[c, 0]]  = A
                    Bc[pixel[c, 0]]  = B
                    Cc[pixel[c, 0]]  = C
                    Ncc[pixel[c, 0]] = B
                    
                F[:, pixel[:, 0]] = a0
                varO[pixel[:, 0]] = er
                POM_NBF_FIT[pixel[:, 0]] = Nbfit


            yfit = im[longueur, pix_isat]
            Alpha = alpha[pix_isat]

            A = Alpha[:, 0] * ((d + offset + i + offset) / 2)
            B = 1
            C = - np.sum(yfit, axis=1) / (i - d + 1)

            delta = B * B - 4 * A * C
            
            a0 = (-B + np.sqrt(delta)) / (2 * A)

            A1 = Alpha[:, 0] * (a0 ** 2)
            A1 = np.repeat(A1[:, np.newaxis], len(longueur), axis=1)
            Ek = yfit - A1 * (longueur + offset)
            er = np.var(Ek, axis=1) / (i - d + 1)
            
            Ac[pix_isat[:, 0]]  = A
            Bc[pix_isat[:, 0]]  = B
            Cc[pix_isat[:, 0]]  = C
            Ncc[pix_isat[:, 0]] = i - d + 1
            
            F[:, pix_isat[:, 0]] = a0
            varO[pix_isat[:, 0]] = er
            POM_NBF_FIT[pix_isat[:, 0]] = i - d + 1
        
        elif i>=0: # cas ou l'on a pas assez de points pour faire un calcul mais plus d'une frame à disposition
            pix_isat = np.intersect1d(np.argwhere(indSat == i), PixaTraiter)
            a = framesat[pix_isat]
            pix_isat = pix_isat[:, np.newaxis]
            true_framesat = framesat[pix_isat] + offset
            S_adu = sat[pix_isat] / (true_framesat - 0.5)
            F[:, pix_isat[:, 0]] = np.transpose(S_adu)
            Stdev = 0.5 * sat[pix_isat] / (true_framesat**2 - true_framesat)
            varO[pix_isat[:, 0]] = np.transpose(Stdev)**2
            POM_NBF_FIT[pix_isat[:, 0]] = a
            Ac[pix_isat[:,0]] = np.nan
            Bc[pix_isat[:,0]] = np.nan
            Cc[pix_isat[:,0]] = np.nan
            Ncc[pix_isat[:,0]] = a

    pix_isat = np.intersect1d(np.argwhere(framesat == 1), PixaTraiter) # cas où il n'y a qu'une seule frame pour faire le calcul (saturation dès la première frame)
    # pix_isat = np.intersect1d(np.argwhere(indSat == i), PixaTraiter)
    pix_isat = pix_isat[:, np.newaxis]
    true_framesat_1 = 1 + offset
    S_adu = sat[pix_isat] / (true_framesat_1 - 0.5)
    F[:, pix_isat[:, 0]] = np.transpose(S_adu)
    if true_framesat_1 == 1:
        varO[pix_isat[:, 0]] = np.transpose(sat[pix_isat])**2
    else:
        Stdev = 0.5 * sat[pix_isat] / (true_framesat_1**2 - true_framesat_1)
        varO[pix_isat[:, 0]] = np.transpose(Stdev)**2
    POM_NBF_FIT[pix_isat[:, 0]] = 1
    Ac[pix_isat[:,0]] = np.nan
    Bc[pix_isat[:,0]] = np.nan
    Cc[pix_isat[:,0]] = np.nan
    Ncc[pix_isat[:,0]] = 1
    
    B = F[0, :]
    varFlux = varO
    return (B, varFlux, carteCos, POM_NBF_FIT, Ac, Bc, Cc, Ncc)
def FitCosmic_old(image, indSat, d, alpha, sat, framesat, CR_VARJ, CR_VMRJ, CR_THRJ, NBFRMIN, indv):
    PixaTraiter = indv
    cos = []
    im = image.copy()
    madLim = np.zeros(2048 * 2048)
    Med = np.zeros(2048 * 2048)
    vr = np.zeros(2048 * 2048)
    fr = np.zeros(2048 * 2048)
    F = np.zeros([1, 4194304])  # matrice des résultats du fit
    varO = np.zeros(4194304)  # variance sur l'offset
    carteCos = np.zeros(2048 * 2048)
    POM_NBF_FIT = np.zeros(2048 * 2048)

    # on prend ici un coef de non linéarité à partir d'un fichier de carac  / ou le fichier test pour ratir
    
    Ac, Bc, Cc, Ncc = [np.zeros(2048 * 2048) for i in range(4)]
    
    for i in np.unique(indSat):
        if i - NBFRMIN >= 0:
            pix_isat = np.intersect1d(np.argwhere(indSat == i), PixaTraiter)
            pix_isat = pix_isat[:, np.newaxis]
            longueur = (np.arange(d-1, i)).astype(np.int64)
            y1 = im[longueur, pix_isat]

            # selection des pixels non eratics et var/flux ok
            v = np.var(y1, axis=1)
            vr[pix_isat[:, 0]] = v

            f = np.mean(y1, axis=1)
            fr[pix_isat[:, 0]] = f

            CR = np.argwhere(v > CR_VARJ)  # 300 #calibration : 120
            g = np.argwhere(v / f > CR_VMRJ)  # 7 #calibration : 2
            indCR = np.intersect1d(g, CR)
            pix = pix_isat[indCR]
            y = im[longueur, pix]

            # calcul de la limite à partir de la mediane et de la MAD
            med = np.median(y, axis=1)
            med2 = med[:, np.newaxis]
            med2 = np.repeat(med2, np.shape(y)[1], axis=1)
            mad = np.median(np.absolute(y - med2), axis=1)
            madLim[pix[:, 0]] = med + CR_THRJ * mad
            Med[pix[:, 0]] = med

            m = madLim[pix[:, 0]]
            m = m[:, np.newaxis]
            m = np.repeat(m, np.shape(y)[1], axis=1)
            
            cr = np.argwhere(y > m)
            p = cr[:, 0]  # pixels impactés
            t = cr[:, 1]  # temps de l'impact
            

            # selection des cosmics : un seul point au dessus de la limite
            uniq, u = np.unique(p, return_counts=True)
            val = uniq[np.argwhere(u == 1)]
            
            uni = np.argwhere(p == val)

            if np.shape(uni)[0] != 0:

                un = uni[:, 1]
                cr = cr[un]
                tps = cr[:, 1] + d - 1
                p = cr[:, 0]
                pixel = pix[p]

                # on met le temps d'impact dans la carte des pixels impactés
                carteCos[pixel[:, 0]] = tps

                # on supprime le pixel impacté de la liste des pixels normaux
                pix_isat = np.setdiff1d(pix_isat, pixel)[:, np.newaxis]

                # on parcours les cosmics selectionés
                a0 = np.zeros(len(tps))
                er = np.zeros(len(tps))
                Nbfit = np.zeros(len(tps))
                for c in range(0, len(tps)):
                    med = np.nanmedian(im[:, pixel[c, 0]])
                    if med == 0: med = 1
                    l = round(im[tps[c], pixel[c, 0]] / med)
                    FRN_MAXFIT = i
                    FRN1       = d
                    yf = im[longueur, pixel[c, 0]]
                    Al = alpha[pixel[c, 0]]
                    Nc = FRN_MAXFIT - FRN1 + 1
                    A = (Al / 2) * (FRN_MAXFIT**2 - FRN_MAXFIT + 2.*FRN_MAXFIT*l -2.*(tps[c]+1-FRN1+1)*l - FRN1**2 - FRN1)
                    B = Nc - 1
                    C = - np.sum(yf) + im[tps[c], pixel[c, 0]]
                    delta = B * B - 4 * A * C
                    
                    a0[c] = (-B + np.sqrt(delta)) / (2 * A)

                    A1 = Al * (a0[c] ** 2)
                    A1 = np.repeat(A1, len(longueur))
                    Ek = yf - A1 * (longueur+1)
                    er[c] = np.var(np.delete(Ek,tps[c]+1-FRN1)) / (FRN_MAXFIT - FRN1)
                    Nbfit[c] = Nc - 1
                    
                    Ac[pixel[c, 0]]  = A
                    Bc[pixel[c, 0]]  = B
                    Cc[pixel[c, 0]]  = C
                    Ncc[pixel[c, 0]] = B
                    
                F[:, pixel[:, 0]] = a0
                varO[pixel[:, 0]] = er
                POM_NBF_FIT[pixel[:, 0]] = Nbfit


            yfit = im[longueur, pix_isat]
            Alpha = alpha[pix_isat]

            A = Alpha[:, 0] * ((d + i) / 2)
            B = 1
            C = - np.sum(yfit, axis=1) / (i - d + 1)

            delta = B * B - 4 * A * C
            
            a0 = (-B + np.sqrt(delta)) / (2 * A)

            A1 = Alpha[:, 0] * (a0 ** 2)
            A1 = np.repeat(A1[:, np.newaxis], len(longueur), axis=1)
            Ek = yfit - A1 * longueur
            er = np.var(Ek, axis=1) / (i - d + 1)
            
            Ac[pix_isat[:, 0]]  = A
            Bc[pix_isat[:, 0]]  = B
            Cc[pix_isat[:, 0]]  = C
            Ncc[pix_isat[:, 0]] = i - d + 1
            
            F[:, pix_isat[:, 0]] = a0
            varO[pix_isat[:, 0]] = er
            POM_NBF_FIT[pix_isat[:, 0]] = i - d + 1


        
        elif i>=0: # cas ou l'on a pas assez de points pour faire un calcul mais plus d'une frame à disposition
            pix_isat = np.intersect1d(np.argwhere(indSat == i), PixaTraiter)
            a = framesat[pix_isat]
            pix_isat = pix_isat[:, np.newaxis]
            S_adu = sat[pix_isat] / (framesat[pix_isat] - 0.5)
            F[:, pix_isat[:, 0]] = np.transpose(S_adu)
            Stdev = 0.5 * sat[pix_isat] / (framesat[pix_isat]**2 - framesat[pix_isat])
            varO[pix_isat[:, 0]] = np.transpose(Stdev)**2
            POM_NBF_FIT[pix_isat[:, 0]] = a
            Ac[pix_isat[:,0]] = np.nan
            Bc[pix_isat[:,0]] = np.nan
            Cc[pix_isat[:,0]] = np.nan
            Ncc[pix_isat[:,0]] = a

    pix_isat = np.intersect1d(np.argwhere(framesat == 1), PixaTraiter) # cas où il n'y a qu'une seule frame pour faire le calcul (saturation dès la première frame)
    # pix_isat = np.intersect1d(np.argwhere(indSat == i), PixaTraiter)
    pix_isat = pix_isat[:, np.newaxis]
    S_adu = 2 * sat[pix_isat]
    F[:, pix_isat[:, 0]] = np.transpose(S_adu)
    varO[pix_isat[:, 0]] = np.transpose(sat[pix_isat])**2
    POM_NBF_FIT[pix_isat[:, 0]] = 1

    Ac[pix_isat[:,0]] = np.nan
    Bc[pix_isat[:,0]] = np.nan
    Cc[pix_isat[:,0]] = np.nan
    Ncc[pix_isat[:,0]] = 1
    
    B = F[0, :]
    varFlux = varO
    return (B, varFlux, carteCos, POM_NBF_FIT, Ac, Bc, Cc, Ncc)


def CorrectifPersistance(PIM_FRN_SAT, N, DT, amp_1, amp_2, tau_1, tau_2, P_LEVEL, N_PREC):
    TF_RAMP = DT + N * 1.33  # temps à la fin de la rampe

    # persistanceFIN = amp_1 * (1 - np.exp(-TF_RAMP / tau_1)) + amp_2 * (1 - np.exp(-TF_RAMP / tau_2))  # persistance si on avait commencé directement après le premier reset de la rampe prec
    # persistanceDEB = (amp_1 * (1 - np.exp((-DT + 1.33) / tau_1)) + amp_2 * ( 1 - np.exp((-DT + 1.33) / tau_2)))

    persistanceFIN = amp_1 * (1 - np.exp(-TF_RAMP / tau_1)) + amp_2 * (1 - np.exp(
        -TF_RAMP / tau_2))  # persistance si on avait commencé directement après le premier reset de la rampe prec
    persistanceDEB = (amp_1 * (1 - np.exp((-DT) / tau_1)) + amp_2 * (1 - np.exp((-DT) / tau_2)))

    # persistance - persistance accumulée du reset jusqu'au début de l'acquisition
    persistance = persistanceFIN - persistanceDEB

    persistance = np.ravel(persistance)
    # persistance = persistance / (N*1.33) # conversion en e/s pour donner un flux
    persistance = persistance / (10 * N)  # conversion en A/f pour donner un flux compatible avec la carte
    persistance[np.isnan(persistance)] = 0

    Pper = np.intersect1d(np.argwhere(PIM_FRN_SAT * P_LEVEL <= N_PREC), np.argwhere(PIM_FRN_SAT > 0))
    print('Number of pixels impacted by persistence: ', np.shape(Pper)[0])
    map_persistance = np.zeros(2048 * 2048)
    map_persistance[Pper] = persistance[Pper]
    map_persistance[np.isnan(map_persistance)] = 0

    return (map_persistance)


def SaveFit(image, nombreFrame, noms_entete, nom_fit, HEAD, Commentaire,overwrite=True):
    #hdr = fits.Header()
    
    hdr = fits.open(Commentaire)[0].header

    hdr[HEAD] = Commentaire
    primary = fits.PrimaryHDU(header=hdr)

    image_hdu = fits.ImageHDU(image[0], name=noms_entete[0], header=hdr)
    hdul = fits.HDUList([primary, image_hdu])
    hdul[1].header.append('measure')
    hdul[1].header['measure'] = noms_entete[0]

    n = np.shape(image)[0]

    if nombreFrame != 1:
        for i in range(2, n + 1):
            hdul.append(fits.ImageHDU(image[i - 1], name=noms_entete[i-1]))
            hdr = hdul[i].header
            hdr.append('measure')
            hdul[i].header['measure'] = noms_entete[i - 1]

    nom = str(nom_fit) + '.fits'
    hdul.writeto(nom, overwrite=overwrite)


def Save(image, name, HEAD, Commentaire, type):
    hdu = fits.PrimaryHDU()
    hdu.data = image.astype(type)

    for i in range(len(HEAD)):
        hdu.header.set(HEAD[i], Commentaire[i])

    nom = str(name) + '.fits'
    hdu.writeto(nom)
