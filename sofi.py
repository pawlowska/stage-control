#@title
import numpy as np
import scipy.stats as st
import math
import pandas as pd
from scipy import signal

from scipy import integrate
from scipy import optimize
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import matplotlib.axes as ax

from gaussfitter import gaussfit
from time import time


# mean exponent for emitter power law distributions of on/off times (from gathered data)
m_on = 1.4707 
std_on = 0.0071
m_off = 1.4022
std_off = 0.0070

# ^^^ PARMS FOR SIMULATION. 


def fft_cov(list1, list2, lag=0):
    N=np.shape(list2)[0]
    if lag==0:
        cov=signal.fftconvolve(list1, list2[::-1], mode='valid')/len(list2)-np.array(signal.fftconvolve(list1,np.ones(len(list2)), mode='valid'))*np.array(signal.fftconvolve(np.ones(len(list1)),list2[::-1], mode='valid'))/(len(list1)**2)
    else:
        cov=np.array(signal.fftconvolve(list1, list2[:N-(lag+1)][::-1], mode='valid')[1:lag+1])/(len(list2[:N-(lag+1)]))-np.array(signal.fftconvolve(list1,np.ones(len(list2[:N-(lag+1)])), mode='valid'))[1:lag+1]*np.array(signal.fftconvolve(np.ones(len(list1)),list2[:N-(lag+1)][::-1], mode='valid'))[1:lag+1]/(len(list2[:N-(lag+1)])**2)
        
    return cov

def load_asc_movie(fname):
    """Loads asc file from Andor Solis data. Presents as list with dims: (frames, resolution, resolution).
    
    Args: 
    
    """
    
    movie = []
    frame = []
    acquisition = []
    j = 0
    roi_size = None
    with open(fname) as f:
        for line in f:
            l = line.split("\t")
            if len(l) < 20:
                acquisition.append(line)
                #print(len(l))
                #print(l)
            else:
                if roi_size == None:
                    roi_size = len(l)-2
                frame.append(l[1:-1])
                j += 1
                if j == roi_size:
                    movie.append(np.array(frame, dtype=np.float64))
                    j = 0
                    frame = []
                    
        return movie, acquisition


def one_gauss(x, *args):
    
    a, m, s, c = args
    
    return a * st.norm.pdf(x, loc=m, scale=s) + c


def fit_one_gauss(data, init):
    
    x = np.linspace(0, len(data)-1, len(data))
    popt, pcov = curve_fit(one_gauss, x, data, p0=init)
    stds = np.sqrt(np.diag(pcov))

    return popt, stds


def double_gauss(x, *args):
    """Creates 2D gaussian distribution.
  
    Args:
        x: linear space, resolution same as data frames
        *args: tuple of a1, a2, m1, m2, s1, s2, c
        a1, a2: amplitudes 
        m1, m2: means
        s1, s2: standard deviations
        c: background for whole image
      
    Returns:
        2D array
    """
  
    a1, a2, m1, m2, s1, s2, c = args
    ret = a1 * st.norm.pdf(x, loc=m1, scale=s1)
    ret += a2 * st.norm.pdf(x, loc=m2, scale=s2)
    ret += c
  
    return ret

def triple_gauss(x, *args):
    """Creates 2D gaussian distribution.
  
    Args:
        x: linear space, resolution same as data frames
        *args: tuple of a1, a2, a3, m1, m2, m3, s1, s2, s3, c
        ai: amplitudes 
        mi: means
        si: standard deviations
        c: background for whole image
      
    Returns:
        2D array
    """
  
    a1, a2, a3, m1, m2, m3, s1, s2, s3, c = args
    ret = a1 * st.norm.pdf(x, loc=m1, scale=s1)
    ret += a2 * st.norm.pdf(x, loc=m2, scale=s2)
    ret += a3 * st.norm.pdf(x, loc=m3, scale=s3)
    ret += c
  
    return ret


def fit_double_gauss(data, init):
    """Fits gaussian function to chosen cross section of image

    Args:
        data: 1d array, chosen cross section of image
        init: *args as tuple a1, a2, m1, m2, s1, s2, c

    Returns:
        1d array of optimized values of parameters a1, a2, m1, m2, s1, s2, c
    """

    x = np.linspace(0, len(data)-1, len(data))
    popt, pcov = curve_fit(double_gauss, x, data, p0=init)
    stds = np.sqrt(np.diag(pcov))

    return popt, stds


def fit_triple_gauss(data, init):
    """Fits gaussian function to chosen cross section of image

    Args:
        data: 1d array, chosen cross section of image
        init: *args as tuple a1, a2, a3, m1, m2, m3, s1, s2, s3, c

    Returns:
        1d array of optimized values of parameters a1, a2, a3, m1, m2, m3, 
        s1, s2, s3, c
    """

    x = np.linspace(0, len(data)-1, len(data))
    popt, pcov = curve_fit(triple_gauss, x, data, p0=init)
    stds = np.sqrt(np.diag(pcov))

    return popt, stds


def get_times(m_on, m_off, frame_number, how_much, how_precise=0.3):    
    """Creates a series of alternating times when emitter is on and off.
    
        Args:
            m_on, m_off: floats, parameters in distribution of times on and off
            how_precise: float, the smallest part of frame emitter can be on/off for, 
            e.g. 0.3
            how_much: float, the longest time (in frames) emitter can be on/off, e.g. 100
            
        Returns:
            times: 1D array of alternating on/off times
            """

    x_on = lambda x: (1/x)**m_on
    x_off = lambda x: (1/x)**m_off
    integral_on = integrate.quad(x_on, how_precise, how_much)
    integral_off = integrate.quad(x_off, how_precise, frame_number)

    
    class on_gen(st.rv_continuous):
        def _pdf(self,x):
            return ((1/x)**m_on)/integral_on[0]  # Normalized over its range


    class off_gen(st.rv_continuous):
        def _pdf(self,x):
            return (1/x)**m_off/integral_off[0]


    on = on_gen(a=how_precise, b=how_much, name='on')  # on times experience 
                                                       # cutoff at a certain point
    off = off_gen(a=how_precise, name='off')  # off times dont have cutoff

    l = frame_number  # to ensure the vector is long enough to 
                      # translate into intensities
    times_on = on.rvs(size=l)
    times_off = off.rvs(size=l)
    times = np.zeros(2*l)
    
    
    for i in range(l):
        times[2*i] += times_on[i]
        times[2*i+1] += times_off[i]
    
    return np.array(times) 
    
        
def frames_intensity(times, frames, precision):
    """Creates series of floats from 0 to 1 corresponding to fraction of frame when emitter
    was turned on.
    
    Args:
        times: 1d array from get_times
        frames: int, number of frames in film
        precision: int, number of significant figures in fractions
        
    Returns:
        intensity: 1d array (length same as number of frames)
    """

    changes0 = [np.sum(times[:i]) for i in range(1,len(times)+1)]
    changes = [round(ch, precision) for ch in changes0]
    intensity = []
    frame_number = 1
    I = 0
    
    for i in range(len(changes)):
        if changes[i]>=frame_number:
            rest = changes[i] - frame_number
            split = times[i] - rest
            if i%2 == 0:
                I += round(split, precision)
                intensity.append(I)
                I = 0
            else:
                intensity.append(I)
                I = 0
            while rest>1:
                if i%2 == 0:
                    intensity.append(1)
                else:
                    intensity.append(0)
                rest -= 1
            if i%2 == 0:
                I += round(rest, precision)
                if i == len(changes)-1:
                    intensity.append(I)
            frame_number = int(changes[i]//1+1)
        else:
            if i%2 == 0:
                I += round(times[i], precision)
        
    #print(len(intensity))            
    return np.array(intensity[:frames]) 


def make_frame(xo, yo, sep, sigma, photon_count, intensity, res, k, fi):
    """Creates image of photon count distribution for five emitters 
    illuminated by cosine pattern (or not if k, fi = 0). Photon count 
    randomized by Poisson distribution with average of input photon_count.

    Args:
        xo, yo: position of corner emitter
        sep: separation between emitters in vertical
        sigma: standard deviation, same for all emitters
        photon_count: average total photon count per frame per one emitter
        intensity: 1d array of 5 floats valued from 0 to 1 representing fraction 
            of frame during which emmiter was on
        res: resolution of image
        k = [kx, ky]: wave vector for illuminating pattern
        fi: float, phase shift of illuminating pattern

    Returns:
        2D array of photon count distribution
    """
    
    # background shot noise around 1% of emitter signal on one px
    mean_n = 0.01*photon_count
    d3 = np.random.poisson(mean_n, size=(2*res, 2*res))
    # bigger frame in case of getting signal far from the center


    # center coordinates translated into bigger frame
    cov = [[sigma**2,0], [0,sigma**2]]
    mean = [[xo + i*sep + res//2, yo + res//2] for i in range(3)]
    mean2 = [[xo + res//2, yo + int((i+1)*sep*0.8) + res//2] for i in range(2)]
    phot = np.random.poisson(photon_count, 5)
    
    
    for i in range(3):
        if intensity[i]>0:
            x = k[0] * mean[i][0] + k[1] * mean[i][1] + fi
            photons = abs(int(intensity[i]*phot[i]*np.cos(x)))
            d = np.random.multivariate_normal(mean[i], cov, photons).astype(np.int16)
            unique, counts = np.unique(d, return_counts=True, axis=0)
            idx = np.array(unique[:, 0]), np.array(unique[:, 1])
            d3[idx] += counts
            
            
    for i in range(2):
        if intensity[i+3]>0:
            x = k[0] * mean2[i][0] + k[1] * mean2[i][1] + fi
            photons = abs(int(intensity[i+3]*phot[i+3]*np.cos(x)))
            d = np.random.multivariate_normal(mean2[i], cov, photons).astype(np.int16)
            unique, counts = np.unique(d, return_counts=True, axis=0)
            idx = np.array(unique[:, 0]), np.array(unique[:, 1])
            d3[idx] += counts
     
    return d3[res//2 : 3*res//2, res//2 : 3*res//2]


def make_frame2(xo, yo, sep, sigma, photon_count, intensity, res, k, fi):
    
    d3 = np.random.normal(loc=mean_n, scale=std_n, size=(2*res, 2*res))
    cov = [[sigma**2,0], [0,sigma**2]]
    mean = [[xo + i*sep, yo + int((i+1)*sep*0.8)] for i in range(10) for j in range(10)]
    phot = np.random.poisson(photon_count, 100)
    
    
    for i in range(3):
        if intensity[i]>0:
            x = k[0] * mean[i][0] + k[1] * mean[i][1] + fi
            photons = abs(int(intensity[i]*phot[i]*np.cos(x)))
            d = np.random.multivariate_normal(mean[i], cov, photons).astype(np.int16)
            unique, counts = np.unique(d, return_counts=True, axis=0)
            idx = np.array(unique[:, 0]), np.array(unique[:, 1])
            d3[idx] += counts
            
            
    for i in range(2):
        if intensity[i+3]>0:
            x = k[0] * mean2[i][0] + k[1] * mean2[i][1] + fi
            photons = abs(int(intensity[i+3]*phot[i+3]*np.cos(x)))
            d = np.random.multivariate_normal(mean2[i], cov, photons).astype(np.int16)
            unique, counts = np.unique(d, return_counts=True, axis=0)
            idx = np.array(unique[:, 0]), np.array(unique[:, 1])
            d3[idx] += counts
     
    return d3[res//2 : 3*res//2, res//2 : 3*res//2]
    

def make_film(xo, yo, sep, sigma, photon_count, frame_number, res, k=[0,0], fi=0):
    """Creates sequence of images of photon count distribution for five emitters
    illuminated by excitation light pattern of cosine.

    Args:
        frame_number: number of frames. frame rate = ?
        rest same as in make_frame

    Returns:
        1D array of frames
    """
    # drawing exponents for on/off times (each emitter has
    # in general a different exponent)
    m_ons = st.norm.rvs(loc=m_on, scale=std_on, size=5)
    m_offs = st.norm.rvs(loc=m_off, scale=std_off, size=5)
    # calculating how much time each emitter is on/off for
    times_all = [get_times(m_ons[i], m_offs[i], frame_number,
                           how_much=int(frame_number/10)) for i in range(5)]
    # translating the times above into intensities on each frame
    intensity = np.vstack([frames_intensity(times, frame_number, 2) 
                           for times in times_all])
    intensity_t = np.transpose(intensity)
    lst = []

    for i in range(frame_number):
        lst.append(make_frame(xo, yo, sep, sigma, photon_count, intensity_t[i], res, k, fi))

    return np.array(lst)



def fluctuation(film, x, y):
    """Creates signal fluctuation function for a given pixel (x, y) in time,
    counted in frame rate units.

    Args:
        film: 1D ARRAY of frames from make_film
        x, y: coordinates of chosen pixel

    Returns:
        1D array of values of photon count fluctuations at chosen pixel
    """

    return film[:, x, y]
def autocumulant_2(film, lag=0):
    """Calculates second order autocumulant for a series of image frames.
    For lag!=0 gives value of sum of time-lagged autocum.

    Args:
        film: 1D ARRAY of frames from make_film
        lag: int, number of frames two intensity traces are shifted 
        in regard to one another

    Returns:
        SOFI image: 2D array of autocumulant values
    """
    
    #time0 = time()
    N = film.shape[0]
    res1 = film.shape[1]
    res2 = film.shape[2]
    d_C2=[[np.sum(fft_cov(film[:,i,j], film[:,i,j], lag)) for j in range(res2)] for i in range(res1)]
               
        #print(time()-time0)
    
    return d_C2

def cov_autocumulant_2(film, lag=0):
    """Calculates second order autocumulant for a series of image frames.
    For lag!=0 gives value of sum of time-lagged autocum.

    Args:
        film: 1D ARRAY of frames from make_film
        lag: int, number of frames two intensity traces are shifted 
        in regard to one another

    Returns:
        SOFI image: 2D array of autocumulant values
    """
    
    #time0 = time()
    N = film.shape[0]
    res = film.shape[1]
    
    if lag==0:
        d_C2 = [[np.var(film[:,i,j]) for j in range(res)] for i in range(res)]
        
    else:
        d_C2 = np.zeros((res,res))
        for a in range(1, lag+1):  # no autocorr.
            d = [[np.cov(film[a:,i,j], film[:N-a,i,j])[0,1] 
                  for j in range(res)] for i in range(res)]
            d_C2+=d
        #print(time()-time0)
    
    return d_C2


def plot_corr_in_time(film, x, y, length=None, Fourier=0):
    # dt: time interval in secs0,000 (frame length)
    # Fourier=0: doesnt show fourier, Fourier!=0: shows fourier
    
    N, h, w = np.shape(film)
    mean_v = [np.cov(film[a:,x,y], film[:N-a,x,y])[0,1]  for a in range(N)]
 
    if length==None:
        plt.plot(mean_v)
    else:
        plt.plot(mean_v[:length])
    plt.title("AC2 signal in time for point (%i, %i)" %(x,y))
    plt.show()
    if Fourier!=0:
        F = np.fft.fft(mean_v)
        dt = input("input frame length: ")
        freqs = np.fft.fftfreq(len(F),float(dt))  # signal frequency in Hz
        plt.plot(freqs,np.abs(F), '.')
        plt.title("Fourier transform")
        plt.show()
    
    return mean_v


def res_increase_std(val1, dval1, val2, dval2, px1=1, px2=1):
    "Standard deviation for value val1/val2"
    return np.sqrt((dval1/val2)**2 + (val1*dval2/(val2**2))**2)*(px1/px2)


def autocumulant_3(film, lag1=0, lag2=0):
    """Calculates third order autocumulant for a series of image frames.
    For lag1,lag2!=0 gives value of sum of time-lagged autocum.

    Args:
        film: 1D ARRAY of frames from make_film
        lag1, lag2: ints, number of frames three intensity traces are
        shifted in regard to one another

    Returns:
        SOFI image: 2D array of autocumulant values
    """
    
    
    N = film.shape[0]
    res1 = film.shape[1]
    res2 = film.shape[2]
    
    if lag1==0 and lag2==0:
        d_C3 = [[(np.mean((film[:,i,j] - np.mean(film[:,i,j]))**3)) 
             for j in range(res2)] for i in range(res1)]
        
    elif np.max((lag1,lag2))>0 and lag1*lag2==0:
        lag = np.max((lag1,lag2))
        d_C3 = np.zeros((res1,res2))
        for a in range(1, lag+1):  # no autocorr.
            d = [[(np.mean(np.var(film[a:,i,j])*(film[:N-a,i,j]-np.mean(film[:N-a,i,j]))))
                  for j in range(res2)] for i in range(res1)]
            d_C3+=d
            
    else:
        d_C3 = np.zeros((res1,res2))
        for a in range(1, lag1+1):
            for b in range(1, lag2+1):
                tb = np.max((a,b))
                ts = np.min((a,b))
                d = [[(np.mean((film[tb:,i,j]-np.mean(film[tb:,i,j]))*
                      (film[tb-ts:N-ts,i,j]-np.mean(film[tb-ts:N-ts,i,j]))*
                      (film[:N-tb,i,j]-np.mean(film[:N-tb,i,j]))))
                      for j in range(res2)] for i in range(res1)]
                d_C3+=d
    
    
        # Cumulants may be negative, depending on fluctuation
        # pattern, so images need to be shown for absolute value.

    return d_C3



def autocumulant_4(film, lag1=0, lag2=0, lag3=0):
    """Creates fourth autocumulant signal image for a set of frames.

    Args:
        film: 1D array of frames from make_film
        res: int, resolution of image

    Returns:
        SOFI image: 2D array of autocumulant values
    """
    
    N, res, res2 = np.shape(film)
    if lag1==0 and lag2==0 and lag3==0:
        d_C4 = [[(np.mean((film[:,i,j] - np.mean(film[:,i,j]))**4) - 3 * np.var(film[:,i,j])) for j in range(res2)] for i in range(res)]
    else:
        d_C4 = np.zeros((res,res2))
        for a in range(1, lag1+1):
            for b in range(1, lag2+1):
                for c in range(1, lag3+1):
                    tb = np.max((a,b,c))
                    ts = np.min((a,b,c))
                    tt = int(a*b*c/(ts*tb))
                    d = [[(np.mean((film[tb:,i,j]-np.mean(film[tb:,i,j]))*
                          (film[tb-ts:N-ts,i,j]-np.mean(film[tb-ts:N-ts,i,j]))*
                          (film[tb-tt:N-tt,i,j]-np.mean(film[tb-tt:N-tt,i,j]))*
                          (film[:N-tb,i,j]-np.mean(film[:N-tb,i,j]))) 
                          - np.cov(film[tb:,i,j],film[tb-ts:N-ts,i,j])[0,1]-np.cov(film[tb:,i,j],film[tb-tt:N-tt,i,j])[0,1]-np.cov(film[tb-ts:N-ts,i,j],film[tb-tt:N-tt,i,j])[0,1])
                          for j in range(res2)] for i in range(res)]
                    d_C4+=d

        # Cumulants may be negative, depending on fluctuation
        # pattern, so images need to be shown for absolute value.

    return d_C4


def pixel_covariance(film, x, y, r):
    """Calculates covariance signal for chosen pixel (x,y) from surrounding
    pixels within radius r. Pixel must be at least in distance r away from both 
    edges of input image.
    
    Args:
        film: 3d tensor of image fluctuation in time
        x, y: pixel coordinates
        r: radius of area around pixels, r = int(w0 * sqrt(2))
        
    Returns:
        cov_total: float giving value of covariance signal at pixel
    """
    
    N, h, w = film.shape
    
    
    # only pixels r away from edges are ok, rest is being considered 
    # irrelevant for output image
    if x < r or x-r > h or y < r or y-r > w:
        print("Neighbourhood extends image limits")
        return -1
    
    
    # list of all shifts around centre within distance r:
    xi, yi = [i for i in range(-r, r+1)], [i for i in range(-r, r+1)]
    
    
    # list of possible coordinates shifts so that shifted pixel 
    # is within radius r:
    pairs = [(i,j)  for i in xi for j in yi if i**2+j**2<=r**2] 
    
    
    # list of pairs of pixels around (x,y) which covariance signal 
    # is calculated from. division by 2: earlier we have repeating sequences
    # of shifts; we dont have //2 + 1 because we don't want to calculate
    # autocumulant signal for pixel in middle (then shot noise is not reduced).
    pairmap = [[[x+el[0], x-el[0]], [y+el[1], y-el[1]]] for el in pairs[:len(pairs)//2]] 
    
    
    # complete surrounding of pixel (x,y)
    sur = [film[el] for el in pairmap]
    
    
    covs = [(el[0] - np.mean(el[0])) * (el[1] - np.mean(el[1])) for el in sur]
    cov_total = np.sum(covs)/(N-1.)
    
    return cov_total
  

def crosscumulant_2(film, w0):
    """Crosscumulant function of order 2. 
    
    Args:
        film: 3d tensor of image fluctuation in time
        w0: FWHS of PSF
        
    Returns:
        xcum2: 2d array of crosscumulant signal of 2nd order
    """
    
    N, h, w = film.shape
    res = h
    radius = 2*int(w0 * np.sqrt(2)//2)
    d_tensor = np.zeros((2*res, 2*res, N))
    xcum2 = np.zeros((2*res, 2*res))
    
    # image englarged by factor of 2
    for i in range(res):
        for j in range(res):
            d_tensor[2*i][2*j] = fluctuation(film, i, j)
            d_tensor[2*i+1][2*j] = fluctuation(film, i, j)
            d_tensor[2*i][2*j+1] = fluctuation(film, i, j)
            d_tensor[2*i+1][2*j+1] = fluctuation(film, i, j)
    
            
    # range of pixels such that they are radius away from edges       
    for i in range(radius, 2*res-radius):
        for j in range(radius, 2*res-radius):
            xcum2[i][j] = pixel_covariance(d_tensor, i, j, radius)

    return xcum2
    
    
def blinking_pixel_batch(film, pairs, i, batch):
    """Fluctuation pattern for ith blinking point in pairs, mean over batch-frame batches."""
    
    frames = np.shape(film)[0]
    flucts = fluctuation(film, int(pairs[i][0]), int(pairs[i][1]))
    xo = int(pairs[i][0])
    yo = int(pairs[i][1])
    
    on = [i for j in range(frames//batch) for i in range(0+j*batch, (j+1)*batch) 
          if flucts[i]>np.mean(flucts[0+j*batch:(j+1)*batch])]
    blink = np.zeros(frames)
    
    for i in range(len(on)):
        blink[on[i]] = 1

     
    """plt.figure(figsize=(20,5))
    plt.subplot(2, 1, 1)
    plt.title("Natężenie punktu (%i,%i) w czasie" %(xo, yo), fontsize=17)
    plt.plot(flucts)
    plt.ylabel("Natężenie na klatce", fontsize=14)
    plt.subplot(2, 1, 2)
    plt.plot(blink)
    plt.xlabel("Liczba klatek", fontsize=14)"""
    
    
    return blink


def times_on_and_off(vector):
    """Creates lists of on and off periods counted in frames.
    
    Args:
        vector: 1D array of 1/0 values and length of number of frames.
                Represents when emitter is on or off.
                
    Returns:
        tuple of lists of on and off periods."""
    
    times_on, times_off = [], []
    time1, time0 = 0, 0

    for i in range(len(vector)):
        if vector[i] == 1:
            time1 += 1
            if time0 != 0:
                times_off.append(time0)
                time0 = 0
        else:
            time0 += 1
            if time1 != 0:
                times_on.append(time1)
                time1 = 0
                
    return times_on, times_off

def power_law(x, *args):
    """Creates 2D gaussian distribution.
  
    Args:
        x: linear space, resolution same as data frames
        m: power
      
    Returns:
        2D array
    """
  
    m, A = args
    ret = A/((x)**(m))
  
    return ret


def fit_power_law(data, init):
    """Fits power law function to histogram of on/off times.

    Args:
        data: 1d array, chosen cross section of image
        init: *args as tuple a1, a2, m1, m2, s1, s2, c`

    Returns:
        1d array of optimized values of parameters a1, a2, m1, m2, s1, s2, c
    """
    
    x = np.linspace(1, len(data), len(data))
    #sigma_in = np.sqrt(data)
    #sigma_in[sigma_in==0]=0.01
    #popt, pcov = curve_fit(power_law, x, data, p0=init, sigma=sigma_in)
    popt, pcov = curve_fit(power_law, x, data, p0=init)
    perr = np.sqrt(np.diag(pcov)) 

    return popt, pcov


def m_histograms(film, centers, batch_size, data_path):
    
    m2 = [blinking_pixel_batch(np.array(film), centers, i, batch_size) for i in range(len(centers))]
    
    OUT0, OUT1 = _get_m_values(m2, data_path)
    data = (OUT0, OUT1)
    name = ('m_off', 'm_on')
    
    for i in range(2):
        plt.figure(figsize=(10,8))
        n, bins, patches = plt.hist(data[i], bins=10)
        plt.title("Histogram wartości parametru %s" %name[i], fontsize=17)
        plt.xlabel("Wartość parametru", fontsize=15)
        plt.ylabel("Liczba otrzymanych wartości parametru", fontsize=15)


        W, p = st.shapiro(data[i])
        D, p2 = st.kstest(data[i], 'norm', args=(np.mean(data[i]),np.std(data[i],ddof=1)))


        print(name[i], '\n', 'mean: ', np.mean(data[i]) )
        print('standard deviation of mean:', np.std(data[i], ddof=1)/np.sqrt(len(data[i])))
        print('skewness:', st.skew(data[i]))
        print('shapiro:', p)
        if p>=0.05:
            print('zatem na poziomie istotnosci 5% dane sa z gaussa')
        print('k-s:', p2)
        if p2>=0.05:
            print('zatem na poziomie istotnosci 5% dane sa z gaussa', '\n')

            
def _get_m_values(m2, data_path):

    ON = [times_on_and_off(m)[0] for m in m2]
    OFF = [times_on_and_off(m)[1] for m in m2]

    binz = [i+0.5 for i in range(10)]

    N0 = [np.histogram(off, bins=binz)[0] for off in OFF]
    N1 = [np.histogram(on, bins=binz)[0] for on in ON]

    x = binz[:(len(binz)-1)]
    init = 1.46, 10

    OUT0 = [fit_power_law(n0, init)[0][0] for n0 in N0]
    OUT1 = [fit_power_law(n1, init)[0][0] for n1 in N1]

    S_0 = [np.sqrt(fit_power_law(n0, init)[1][0,0]) for n0 in N0]
    S_1 = [np.sqrt(fit_power_law(n1, init)[1][0,0]) for n1 in N1]

    np.savetxt(data_path+"m_ons.txt", list(zip(OUT1, S_1)))
    np.savetxt(data_path+"m_offs.txt", list(zip(OUT0, S_0)))
    
    return OUT0, OUT1            
            

def get_sigma(data, A, B, sigma, noise):
    """Calculates FWHM of PSF of up to 3 gaussians for cross section of 
    image between two given points.
    
    Args:
        data: 2d array, input image
        A, B: start and finish points for cross section of image in 
            format of int valued list: [x, y]
        sigma: int, input FWHM of PSF in pixels, will be optimized
        noise: int, amount of background photons
        
    Return:
        xsec: 1d array, cross section of image between given points
        sig: optimized parametr(s) for FWHM (float for 1 gaussian,
            tuple of 2 or 3 floats for 2 and 3 gaussians)
    """
    
    from skimage.feature import peak_local_max
    
    
    # Creating cross section
    
    
    Ax, Ay = A
    Bx, By = B
    d = np.sqrt((Ax-Bx)**2 + (Ay-By)**2)
    ABx = np.linspace(Ax,Bx, int(d), dtype=int)
    ABy = np.linspace(Ay,By, int(d), dtype=int)
    profile = [ABx, ABy]
    xsec = data[profile]
    
    
    # Calculating sigmas
    
    
    x = np.linspace(0, len(xsec), len(xsec))
    sig = 0
    out = 0
    many = input("Many peaks? (yes/no) ")
    
    
    if many == 'no':
        max_y = max(xsec)
        max_x = x[xsec.argmax()]
        init = max_y, max_x, sigma, noise
        out, stds = fit_one_gauss(xsec, init)
        #sig = out[2]
        #sig_s = stds[2]
        
        
    elif many == 'yes':
        sep = input("Separation between peaks: ")
        
        if int(sep) <= 0:
            print("Separation must be bigger than zero.")
            
        else:
            peaks = peak_local_max(xsec, min_distance=int(sep))
            if len(peaks) == 2:
                init = (xsec[peaks[0][0]], xsec[peaks[1][0]], peaks[0][0], 
                        peaks[1][0], sigma, sigma, noise)
                out, stds = fit_double_gauss(xsec, init)
                #sig = out[4], out[5]
                #sig_s = stds[4], stds[5]
                
            elif len(peaks) == 3:
                init = (xsec[peaks[0][0]], xsec[peaks[1][0]], xsec[peaks[2][0]], 
                        peaks[0][0], peaks[1][0], peaks[1][0], 
                        sigma, sigma, sigma, noise)
                out, stds = fit_triple_gauss(xsec, init)
                #sig = out[6], out[7], out[8]
                #sig_s = stds[6], stds[7], stds[8]
                
            elif len(peaks) > 3:
                print("Too many peaks detected.")
            else:
                print("Only one peak detected.")
            
    else:
        print("Answer should be yes or no.")
    
    return xsec, out,stds


def photon_finder(data, neighborhood_size, cutoff):
    """
    data_raw: 2D array
    neighbourhood_size: int, minimal space between two emitters
    cutoff: float 
    """    
    
    h, w = np.shape(data)
    res = h
    threshold = data.min() * cutoff
    
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    
    
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2    
        if (x_center<res-2 and x_center>3) and (y_center<res-2 and y_center>3):
            x.append(x_center)
            y.append(y_center)
        
    centers = list(zip(y,x))
        
    plt.figure(figsize=(10,8))
    plt.imshow(data)
    plt.colorbar()
    plt.autoscale(False)
    plt.plot(x, y, 'ro', markersize=1)
    plt.show()

    return centers


def photon_finder_gaussians(image, centers):
    
    #commented for circle=1
    
    param_names = ['angle', 'd_angle', 'offset', 'd_offset', 'amplitude', 'd_amplitude', 'xo', 'd_xo', 
                   'yo', 'd_yo', 'sx', 'd_sx', 'sy', 'd_sy']
    res = {name: [] for name in param_names}
    
    for c in centers:
        xo, yo = tuple(map(int, c))
        fit, cov_p, infodict, errmsg = gaussfit(image[xo-4:xo+4,yo-4:yo+4], circle=0, return_all=1)
        N = np.size(image[xo-4:xo+4,yo-4:yo+4])
        n = len(fit)
        s_sq = (infodict['fvec']**2).sum()/ (N-n) # infodict already contains f(x)-y, which is needed
        cov = cov_p * s_sq
        stds = [np.sqrt(cov[i][i]) for i in range(len(fit))]
        #params = dict(zip(param_names, np.round(fit, 6)))
        params = dict(zip(param_names, np.round(fit, 7)))
        #const, amp, x, y, s, angle = fit
        const, amp, x, y, sx, sy, angle = fit
        x += xo
        y += yo
        res['offset'].append(const)
        res['amplitude'].append(amp)
        res['xo'].append(x)
        res['yo'].append(y)
        #res['s'].append(s)
        res['sy'].append(sy)
        res['sx'].append(sx)
        res['angle'].append(angle)
        res['d_offset'].append(stds[0])
        res['d_amplitude'].append(stds[1])
        res['d_xo'].append(stds[2])
        res['d_yo'].append(stds[3])
        res['d_sx'].append(stds[4])
        res['d_sy'].append(stds[5])
        res['d_angle'].append(stds[6])
        
        
    fits = pd.DataFrame(res)
    #positions = pd.DataFrame(centers, columns=['xo', 'yo'])
        
    return fits


def show_photobleaching(film, guess_A, guess_t, guess_y0, batch):
    """Shows photobleaching curve A*exp(-x*t)+y0 fitted to means of signals taken from batch
    number of frames. Provides fit parameters, requires input guess of parameters.
    
    Args:
        film: 3d ARRAY in format (frames, resoultion, resolution)
        guess_A: amplitude guess
        guess_t: decay constant guess
        guess_y0: offset guess
        batch: amount of frames mean is calculated from    
    """
    
    from scipy.optimize.minpack import curve_fit
    
    # creating data for fit
    N, h, w = film.shape
    mean = []
    for i in range(N//batch):
        mean.append(np.ndarray.mean(film[i*batch:(i+1)*batch]))
        
     
    # fitting exp decay
    x = np.linspace(0, len(mean)+1, len(mean))
    exp_decay = lambda x, A, t, y0: A * np.exp(-x * t) + y0
    
    guess = guess_A, guess_t, guess_y0
    coeff, cov = curve_fit(exp_decay, x, mean, p0=guess)
    A, t, y0 = coeff

    print("A = %s +- %s\nt = %s +- %s\ny0 = %s +- %s" % (A, np.sqrt(cov[0,0]),
                                                         t, np.sqrt(cov[1,1]),
                                                         y0, np.sqrt(cov[2,2])))
    
    
    # plotting data with curve fit
    plt.figure(figsize=(12,8))
    plt.plot(mean, '.')
    plt.plot(A * np.exp(-x * t) + y0, label="A = %s +- %s\nt = %s +- %s\ny0 = %s +- %s" % (A, np.sqrt(cov[0,0]),
                                                         t, np.sqrt(cov[1,1]),
                                                         y0, np.sqrt(cov[2,2])))
    plt.legend(fontsize=14)
    plt.fill_between(x, (A-np.sqrt(cov[0,0])) * np.exp(-x * (t-np.sqrt(cov[1,1]))) + y0-np.sqrt(cov[2,2]),
                    (A+np.sqrt(cov[0,0])) * np.exp(-x * (t+np.sqrt(cov[1,1]))) + y0+np.sqrt(cov[2,2]),
                    alpha=0.5, facecolor='green')

    plt.title("Mean signal in time with fitted curve A * np.exp(-x * t) + y0", fontsize=17)
    plt.xlabel("Frames", fontsize=14)
    plt.ylabel("Intensity", fontsize=14)
    
    



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                           
                           