# module for importing and analysing .spc data for SOFI-ISM purposes
import numpy as np
import matplotlib.pyplot as plt
import sofi
import cv2
from time import time
from scipy import optimize
from scipy import signal
from scipy.interpolate import interp2d

plt.rcParams['image.cmap'] = 'inferno'

def fft_interpolate(image, m=10, mask_name=0):

    fft_image=np.fft.fftshift(np.fft.fft2(image))
    if mask_name!=0:
        mask=np.load('masks\\mask_fft_'+str(mask_name) +'.npy')
        for i in range(fft_image.shape[0]):
            for j in range(fft_image.shape[0]):
                if mask[i][j]:
                    fft_image[i][j]=0   
    fft_image_inter=np.concatenate((np.zeros((int(((fft_image.shape[0]-1)*m-fft_image.shape[0]+1)/2),(fft_image.shape[0]-1)*m+1),dtype='complex128'),np.concatenate((np.zeros((fft_image.shape[0],int(((fft_image.shape[0]-1)*m-fft_image.shape[0]+1)/2)),dtype='complex128'),fft_image,np.zeros((fft_image.shape[0],int(((fft_image.shape[0]-1)*m-fft_image.shape[0]+1)/2)),dtype='complex128')),axis=1),np.zeros((int(((fft_image.shape[0]-1)*m-fft_image.shape[0]+1)/2),(fft_image.shape[0]-1)*m+1),dtype='complex128')),axis=0)
    image_inter=np.fft.ifft2(np.fft.ifftshift(fft_image_inter))
    return image_inter

def load_bio_data(file_info, res=21):
    """ONLY FOR BIO DATA. QD data has different structure and requires
    import via load_data function. Loads and reshapes bio data into multidim tensor. 
    Input data was taken by scanning 9 tiles of 21x21 pixels one after another, so we
    need to glue them toghether into 63x63 image. 
    
    Args:
    n: number of frames
    res: resolution of 1 image tile
    
    Returns:
    scans: array in shape (number of scans, frame number, height, width, detectors)
    vx, vy: translation vectors
    """
    
    """print("Importing tanslation vectors \n======================")
    shiftx = np.array(np.loadtxt(main_path+data_index+"/shiftx.txt", delimiter="\t"))
    shifty = np.array(np.loadtxt(main_path+data_index+"/shifty.txt", delimiter="\t"))

    # translation vectors in pixels
    vx = [shiftx[i]/dx for i in range(14)]
    vy = [shifty[i]/dx for i in range(14)]
    print("Translation vectors successfully imported \n======================")"""

    main_path, data_index, n = file_info
    path2 = main_path+data_index+"/fullx.txt"
    print("Importing bio data from", path2, "\n======================")
    data2 = np.loadtxt(path2, delimiter="\t")
    print("Data successfully imported. Shape of data:", data2.shape, 
         "\n======================")


    s = int(res**2)

    scans = []
    for j in range(14):
        start = n
        end = (s+1)*n-1
        gap = end-start
        x = []
        for i in range(9):
            #print(i, start, end)
            x.append(data2[j,start:end+1])
            start = end + 2*n + 1
            end = start + gap

        x = np.array(x)
        if j==0:
            print("Shape of one detector chunk before reshaping:", x.shape)
        X = x.reshape(3,3,21,21,2000).transpose(4,0,2,1,3)
        Z = X.reshape(2000,63,63).transpose(0,2,1)
        scans.append(Z)
        if j==0:
            print("Final shape of chunk after reshaping:", Z.shape, "\n====================\n")
        """Zmean = np.mean(Z,axis=0)
        plt.imshow(Zmean)
        plt.colorbar()
        plt.show()"""

    scans = np.array(scans)
    # adding extra dimension so it matches non-bio data structure
    scans = np.expand_dims(scans,axis=0)
    scans = scans.transpose(0,1,3,4,2)
    scans = scans.transpose(0,4,2,3,1)
    print("Final shape of bio data (number of scans, frame number, height, width, detectors):", scans.shape)
    
    return scans




def load_spc_data(path, frame_number):
    """Loads and reshapes data into multidim tensor.
    Shape: (number of scans, frame number, height, width, detectors).
    """
    
    data = np.loadtxt(path, delimiter="\t")
    res = int(np.sqrt(np.shape(data)[1]/frame_number))
    #scan = data.reshape((16,res,res,frame_number)).transpose(3,1,2,0)
    scan = data.reshape((16,res,res,frame_number)).transpose(3,2,1,0)
    
    return scan
def fft_cov(list1, list2, lag=0):
    N=np.shape(list2)[0]
    if lag==0:
        cov=signal.fftconvolve(list1, list2[::-1], mode='valid')/len(list2)-np.array(signal.fftconvolve(list1,np.ones(len(list2)), mode='valid'))*np.array(signal.fftconvolve(np.ones(len(list1)),list2[::-1], mode='valid'))/(len(list1)**2)
    else:
        cov=np.array(signal.fftconvolve(list1, list2[:N-(lag+1)][::-1], mode='valid')[1:lag+1])/(len(list2[:N-(lag+1)]))-np.array(signal.fftconvolve(list1,np.ones(len(list2[:N-(lag+1)])), mode='valid'))[1:lag+1]*np.array(signal.fftconvolve(np.ones(len(list1)),list2[:N-(lag+1)][::-1], mode='valid'))[1:lag+1]/(len(list2[:N-(lag+1)])**2)
        
    return cov

def sum_img_t(stack, vx, vy):
    """Produces sum of images translated into place.
    
    Args:
        stack: array of N translated images, dims: (N, h, w)
        vx, vy: vectors if translation in Pythonic x, y
    """

    N, h, w = stack.shape
    im_t = np.zeros((h,w))
    translation_matrix = np.array([[[1,0,x],[0,1,y]] for x,y in zip(vx,vy)])

    
    for i in range(N):
        im = cv2.warpAffine(stack[i], translation_matrix[i], 
                            (w,h))
        im_t += im
        
        
    return im_t

def interpolate(data, step_size=0.1, mode='cubic'):
    x=np.arange(data.shape[0])
    y=np.arange(data.shape[1])
    f=interp2d(x, y, data, kind=mode)
    xx=np.arange(0, data.shape[0]-1, step_size)
    yy=np.arange(0, data.shape[1]-1, step_size)
    data_inter=f(xx,yy)
    
    return data_inter
    

def cross_section(data, A, B):
    """Creates cross section of image from chosen point A to B.
    """
    Ax, Ay = A
    Bx, By = B
    d = np.sqrt((Ax-Bx)**2 + (Ay-By)**2)
    ABx = np.linspace(Ax,Bx, int(d), dtype=int)
    ABy = np.linspace(Ay,By, int(d), dtype=int)
    profile = [ABx, ABy]
    xsec = data[profile]
    
    return xsec

    
    
    
    
def calc_cum(scans, vx, vy, data_index, main_path):
    """Calculates and displays mean image, mean**2 image, 
    2nd autocumulant with timelag 5 and with no timelags as well as
    2nd cross cumulant with timelag=5 and 3rd auto with timelags 5,5.
    
    Args:
        scans: 5d array from load_spc_data. shape: 
        (number of scans, frame number, height, width, detectors)
        vx, vy: translation vectors in pixels (calculated in main)
    """
    
    
    scan_n, N = scans.shape[0], scans.shape[1]
    
    print("Calculating all cumulants for N = %i \n ==============" %N)

    
    # 2nd autocum without timelags
    ac20 = np.array([[sofi.autocumulant_2(scans[j,:,:,:,i]) 
                     for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    # 2nd autocum with lag=5
    ac25 = np.array([[sofi.autocumulant_2(scans[j,:,:,:,i], lag=5) 
                     for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    
    # 3rd autocum with lags<=5
    ac35 = np.array([[sofi.autocumulant_3(scans[j,:,:,:,i], 5,5) 
                     for i in range(scans.shape[4]-2)] for j in range(scan_n)])
  
    # mean
    mean = np.array([[np.mean(scans[j,:,:,:,i], axis=0)
                     for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    # mean squared
    mean_sq = np.array([[np.mean((scans[j,:,:,:,i]), axis=0)**2
                     for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    
    # 2nd crosscum with lag=5
    xc2 = np.array([crosscumulant_2(scans[i,:,:,:,:], vx,vy, 5) 
                    for i in range(scan_n)])

    
    ## images from all detectors shifted into place for chosen scan
    for which_scan in range(scan_n):

        mean_sum = sum_img_t(mean[which_scan,:,:,:], vx, vy)
        mean_sq_sum = sum_img_t(mean_sq[which_scan,:,:,:], vx, vy)
        ac20_sum = sum_img_t(ac20[which_scan,:,:,:], vx, vy)
        ac35_sum = sum_img_t(ac35[which_scan,:,:,:], vx, vy)
        ac25_sum = sum_img_t(ac25[which_scan,:,:,:], vx, vy)
        xc2_pic = xc2[which_scan]

        f1 = plt.figure(figsize=(16,10))
        title = "Comparison for scan "+str(which_scan+1)+" from file "+main_path+data_index+"\n N="+str(N)
        f1.suptitle(title, fontsize=20)
        f1.add_subplot(231)
        plt.imshow(mean_sum)
        plt.colorbar()
        plt.title("Mean", fontsize=15)
        f1.add_subplot(232)
        plt.imshow(mean_sq_sum)
        plt.colorbar()
        plt.title("Mean squared", fontsize=15)
        f1.add_subplot(233)
        plt.imshow(ac20_sum)
        plt.colorbar()
        plt.title("2nd autocumulant \n without timelags", fontsize=15)
        f1.add_subplot(234)
        plt.imshow(ac25_sum)
        plt.colorbar()
        plt.title("2nd autocumulant with \n timelag up to 5 frames", fontsize=15)
        f1.add_subplot(235)
        plt.imshow(ac35_sum)
        plt.colorbar()
        plt.title("3rd autocumulant with \n timelags up to 5 frames", fontsize=15)
        f1.add_subplot(236)
        plt.imshow(xc2_pic)
        plt.colorbar()
        plt.title("2nd crosscumulant with \n timelag up to 5 frames", fontsize=15)
        plt.show()
        
        file_name = data_index+"_"+str(which_scan+1)+"_"+str(N)+".png"
        f1.savefig(file_name)

        
    return f1,mean_sum,mean_sq_sum,ac20_sum,ac35_sum,ac25_sum,xc2_pic


def ac2_one_lag(film, lag):
    """Calculates value for 2nd autocumulant for ONE timelag.
    Useful to see where the correlation function returns only noise.
    """
    
    N = film.shape[0]
    res = film.shape[1]
    d = [[np.cov(film[lag:,i,j], film[:N-lag,i,j])[0,1] 
          for j in range(res)] for i in range(res)]
    
    return d


def calc_lag(scans, lags):
    """Shows correlation for different lags.
    Lags: tuple of ints
    """
    
    f = plt.figure(figsize=(scan_n*6,len(lags)*5))
    for l in range(len(lags)):
        ac2b = np.array([[ac2_one_lag(scans[j,:,:,:,i], lag=lags[l]) 
                 for i in range(scans.shape[4]-2)] for j in range(scan_n)])
        for scan in range(scan_n):
            ac2b_sum = sum_img_t(ac2b[scan,:,:,:], vx, vy)
            f.add_subplot(len(lags),scan_n,scan+l+1)
            plt.imshow(abs(ac2b_sum))
            plt.colorbar()
            plt.title("Correlation for lag of %i frames, scan %i" 
                      %(lags[l], scan+1), fontsize=12) 
def crosscumulant_2(film, vx, vy, lag=0):
    
    #time0=time()
    det = film.shape[3]-2 # number of detectors
    w,h = film.shape[2], film.shape[1]  # shape im img
    N = film.shape[0]
    im_t = np.zeros((h,w))

    
    for x in range(h):
        for y in range(w):
            #time1 = time()
            # d is an array of all possible values of correlations
            # between detector pairs

            if lag==0:
                d = np.array([[np.cov(film[:,x,y,t], film[:,x,y,z])[0,1] 
                              for t in range(det) if t!=z] for z in range(det)])

            else:
                d = np.zeros((det,det-1))
                d=np.array([[np.sum(fft_cov(film[:,x,y,t], film[:,x,y,z],lag))
                        for t in range(det) if t!=z] for z in range(det)])
                #for a in range(1, lag+1):  # no autocorr.
                    #dc = [[np.cov(film[a:,y,x,t], film[:N-a,y,x,z])[0,1] 
                     #      for t in range(det) if t!=z] for z in range(det)]
                   # d += dc

            #print(time()-time1)
            # translation matrix from centre of img
            # it is calculated form (0,0), not from centre!!!
            translation_matrix = np.array([[ [[1,0,(vx[t]+vx[z])/2+x],[0,1,(vy[t]+vy[z])/2+y]]  
                                            for t in range(det) if t!=z] for z in range(det)])

            # d and translation_matrix have shape 14x13!!!

            im = np.sum([[cv2.warpAffine(d[i,j], translation_matrix[i,j], (w,h)) 
                          for i in range(d.shape[0])] for j in range(d.shape[1])],axis=(0,1))

            im_t += im

    #print(time()-time0)

    
    return im_t


def crosscumulant_3(film, vx, vy, lag1=0, lag2=0):
    #film it is a data on which we are calculating XC3
    #shape of the film: (frames, X pixels, Y pixels, detectors + 2 extra useles rows) 
    #vx and vy are an translation vectors of autocumulant's images (14 images obtained on 14 single detector). They are usefull to calculate translation vectors for crosscumulants
    # To calculate cumulant of 3 order we are using 3 light indencity traces. They could be shifted to each others. lag1 and lag2 it is an value of shifts in time (unit - frame duration).
    det = film.shape[3]-2 # number of detectors
    w,h = film.shape[2], film.shape[1]  # shape im img
    N = film.shape[0] #Number of frames
    im_t = np.zeros((h,w)) #our image (in future)

    
    for x in range(h): #loop of x positions
        for y in range(w): #loop of y positions
            
            # d is an array of all possible values of correlations
            # between detector pairs
            # Case when both time lags == 0
            if lag1==0 and lag2==0:
                #AC3(X,Y,Z)=E((X-E(X)*(Y-E(Y)*(Z-E(Z)); E  - expected value
                #t and z and m are a different detectors. What means that X Y Z in these case are idensity traces from different detectors registered in the same time
                
                d = np.array([[[(np.mean((film[:,x,y,t]-np.mean(film[:,x,y,t]))*(film[:,x,y,z]-np.mean(film[:,x,y,z]))*(film[:,x,y,m] - np.mean(film[:,x,y,m])))) for t in range(det) if (t!=z and t!=m)] for z in range(det) if z!=m] for m in range(det)])
            
            
            
            
            
            #Case when both time lags !=0
            #I was to lazy to think about the case when lag 1==0 and lag 2=!0
            if lag1*lag2!=0:
                #tutaj już totalnie się jebłem bo w sumue nie bardzo jest poniższa definicja potrzebna
                
                d = np.zeros((det,det-1,det-2))
                
                for a in range(1, lag1+1):
                    for b in range(1, lag2+1):
                        tb = np.max((a,b))
                        ts = np.min((a,b))
                        
                #Tak jak mówiłem jebłem się tutaj. Ogólnie celem jest stworzyć K=lag1*lag2*det*(det-1)*(det-2) obrazków (det jest liczbą detektorów i wynosi na tą chwilę 14 ale się kiedys zmieni). Każdy obrazek z K obrazków ma w efekcie przypisany swoje dwa time lagi oraz 3 różne detektory. Zasadniczo tutaj na samym początku od razmu możesz zrobić sumowanie po wszystkich time lagach (bez zerowych) dla przypisanych tych trójek samych detektorów. Wtedy już masz M=det*(det-1)*(det-2) obrazków. Każdy z M obrazków jest przesunięty między sobą. By uzyskać jeden zajebisty obrazek musisz już niestety przesuwać je tak jak mi się wydaje że się przesuwa poniżej. 
                        d += [[[np.mean((film[tb:,x,y,t]-np.mean(film[tb:,x,y,t]))*(film[tb-ts:N-ts,x,y,z]-np.mean(film[tb-ts:N-ts,x,y,z]))*(film[:N-tb,x,y,m]-np.mean(film[:N-tb,x,y,m])))for t in range(det) if (t!=z and t!=m)] for z in range(det) if z!=m] for m in range(det)]
             #t and z and m are a different detectors. 
            #X Y Z in these case are idensity traces from different detectors registered not in the same time. There is time lags between traces
                        
            # translation matrix from centre of img
            # it is calculated form (0,0), not from centre!!!

            
            
            
            
            
            #I assume that bellow it is a 3D matrix of shifts value of all images obtained using 3 detectors (for all images from d array). 
            translation_matrix = np.array([[[[[1,0,(vx[t]+vx[z]+vx[m])/3+x],[0,1,(vy[t]+vy[z]+vy[m])/3+y]] for t in range(det) if (t!=z and t!=m)] for z in range(det) if z!=m] for m in range(det)])
           

            # d and translation_matrix have shape 14x13x12!!!
             #I dont know how these exacly works :(
             #these must be function which are shifting all images and summing them in smart way but it is slow
            im = np.sum([[[cv2.warpAffine(d[i,j,k], translation_matrix[i,j,k], (w,h)) for i in range(d.shape[0])] for j in range(d.shape[1])] for k in range(d.shape[2])],axis=(0,1,2))

            im_t += im

    
    
    return im_t

            
def cov_crosscumulant_2(film, vx, vy, lag=0):
    
    #time0=time()
    det = film.shape[3]-2 # number of detectors
    w,h = film.shape[2], film.shape[1]  # shape im img
    N = film.shape[0]
    im_t = np.zeros((h,w))

    
    for x in range(h):
        for y in range(w):
            #time1 = time()
            # d is an array of all possible values of correlations
            # between detector pairs

            if lag==0:
                d = np.array([[np.cov(film[:,y,x,t], film[:,y,x,z])[0,1] 
                              for t in range(det) if t!=z] for z in range(det)])

            else:
                d = np.zeros((det,det-1))
                for a in range(1, lag+1):  # no autocorr.
                    dc = [[np.cov(film[a:,y,x,t], film[:N-a,y,x,z])[0,1] 
                           for t in range(det) if t!=z] for z in range(det)]
                    d += dc

            #print(time()-time1)
            # translation matrix from centre of img
            # it is calculated form (0,0), not from centre!!!
            translation_matrix = np.array([[ [[1,0,(vx[t]+vx[z])/2+x],[0,1,(vy[t]+vy[z])/2+y]]  
                                            for t in range(det) if t!=z] for z in range(det)])

            # d and translation_matrix have shape 14x13!!!

            im = np.sum([[cv2.warpAffine(d[i,j], translation_matrix[i,j], (w,h)) 
                          for i in range(d.shape[0])] for j in range(d.shape[1])],axis=(0,1))

            im_t += im

    #print(time()-time0)

    
    return im_t


def bandpass_filter(signal, freqs, top_cutoff, bottom_cutoff):
    """Bandpass filtering on signal given in Fourier space.
    signal: FFT of signal in time domain
    freqs: 1d array of frequencies
    top_cutoff: float, frequency components above this freq will not pass
    bottom_cutoff: float, frequency components below this freq will not pass
    """

    cut_f_signal = signal.copy()
    cut_f_signal2 = signal.copy()
    cut_f_signal[(abs(freqs)<bottom_cutoff)] = 0
    cut_f_signal2[(abs(freqs)>top_cutoff)] = 0
    c = cut_f_signal + cut_f_signal2
    
    return c


def fit_exp_decay(vector, opt):
    """Fits exponential decay to vector of data wih given
    set of initial parameters.
    vector: 1d vector of input data to fit
    opt: 1d array of size 3 of initial parameters for fit
    opt = amplitude, decay_time, offset
    """
    
    
    x = np.linspace(0,len(vector)-1,len(vector))
    #f = lambda x,A,b,u,fi,t,c: A*(1+b*np.sin(x*u+fi))*np.exp(-t*x)+c
    F = lambda x,A,t,c: A*np.exp(-t*x)+c
    zero = lambda x: x*0

    popt, pcov = optimize.curve_fit(F, x, vector, opt)
    perr = np.sqrt(np.diag(pcov))

    return popt, perr


def test_3_sigma(v1, v2, std1, std2):
    """Performs 3 sigma test for two values.
    """
    m = abs(v1-v2)
    n = 3*np.sqrt(std1**2+std2**2)
    if m<n:
        print("ok, same population")
    else:
        print("not the same population")
def ac2_n(scans, detector, p_x, p_y, time_lags=0):
    A=np.zeros((scans.shape[1],1,1))
    A[:,0,0]=scans[0,:,p_x, p_y, detector]
    ACN=[sofi.autocumulant_2(A[0:i],time_lags)[0][0] for i in range(time_lags, scans.shape[1])]
    return ACN

def ac2_n_shift(scans, detector, p_x, p_y, vx, vy, time_lags=0):
    pxx=p_x+vx[detector]
    pyy=p_y+vy[detector]
    ac_n_sum=np.zeros(scans.shape[1]-time_lags)
    for i in range(1,len(vx)):
        ACN=ac_n(scans, i, int(round(pxx-vx[i], 0)) , int(round(pyy-vy[i], 0)), int(time_lags))
        ac_n_sum=ac_n_sum+ACN
    return ac_n_sum
def ac3_n(scans, detector, p_x, p_y, lag1=0,lag2=0):
    A=np.zeros((scans.shape[1],1,1))
    A[:,0,0]=scans[0,:,p_x, p_y, detector]
    
    ACN=[sofi.autocumulant_3(A[0:i],lag1=0,lag2=0)[0][0] for i in range(np.max((lag1,lag2)), scans.shape[1])]
    return ACN

def ac3_n_shift(scans, detector, p_x, p_y, vx, vy, lag1=0,lag2=0):
    pxx=p_x+vx[detector]
    pyy=p_y+vy[detector]
    time_lags=np.max((lag1,lag2))
    ac_n_sum=np.zeros(scans.shape[1]-time_lags)
    for i in range(1,len(vx)):
        ACN=ac3_n(scans, i, int(round(pxx-vx[i], 0)) , int(round(pyy-vy[i], 0)), lag1, lag2)
        ac_n_sum=ac_n_sum+ACN
    return ac_n_sum
    
    
def rebin(scans, u=0):
    scans2=np.array([np.sum(scans[:,int(u*i):int(u*i+u),:,:,:],axis=1) for i in range(int(scans[0,:,:,:,0].shape[0]/u))])
    return np.transpose(scans2, (1,0,2,3,4))




def ac1_image(scans, vx, vy,):
    scan_n, N = scans.shape[0], scans.shape[1]
    ac5 = np.array([[np.mean(scans[j,:,:,:,i], axis=0) for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    for which_scan in range(scan_n):
        ac5_sum = sum_img_t(ac5[which_scan,:,:,:], vx, vy)
    return ac5_sum

def ac2_image(scans, vx, vy, lag1):
    scan_n, N = scans.shape[0], scans.shape[1]
    ac5 = np.array([[sofi.autocumulant_2(scans[j,:,:,:,i], lag1) for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    for which_scan in range(scan_n):
        ac5_sum = sum_img_t(ac5[which_scan,:,:,:], vx, vy)
    return ac5_sum

def ac3_image(scans, vx, vy, lag1,lag2):
    scan_n, N = scans.shape[0], scans.shape[1]
    ac5 = np.array([[autocumulant_3(scans[j,:,:,:,i], lag1,lag2) for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    for which_scan in range(scan_n):
        ac5_sum = sum_img_t(ac5[which_scan,:,:,:], vx, vy)
    return ac5_sum

def ac4_image(scans, vx, vy, lag1,lag2,lag3):
    scan_n, N = scans.shape[0], scans.shape[1]
    ac5 = np.array([[autocumulant_4(scans[j,:,:,:,i], lag1,lag2,lag3) for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    for which_scan in range(scan_n):
        ac5_sum = sum_img_t(ac5[which_scan,:,:,:], vx, vy)
    return ac5_sum

def ac5_image(scans, vx, vy, lag1,lag2,lag3,lag4):
    scan_n, N = scans.shape[0], scans.shape[1]
    ac5 = np.array([[autocumulant_5(scans[j,:,:,:,i], lag1,lag2,lag3,lag4) for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    for which_scan in range(scan_n):
        ac5_sum = sum_img_t(ac5[which_scan,:,:,:], vx, vy)
    return ac5_sum

def ac6_image(scans, vx, vy, lag1,lag2,lag3,lag4,lag5):
    scan_n, N = scans.shape[0], scans.shape[1]
    ac5 = np.array([[autocumulant_6(scans[j,:,:,:,i], lag1,lag2,lag3,lag4,lag5) for i in range(scans.shape[4]-2)] for j in range(scan_n)])
    for which_scan in range(scan_n):
        ac5_sum = sum_img_t(ac5[which_scan,:,:,:], vx, vy)
    return ac5_sum



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
        #for a in range(1, lag1+1):
            #for b in range(1, lag2+1):
        tb = np.max((lag1,lag2))
        ts = np.min((lag1,lag2))
        d = ([[(np.mean((film[tb:,i,j]-np.mean(film[tb:,i,j]))*
            (film[tb-ts:N-ts,i,j]-np.mean(film[tb-ts:N-ts,i,j]))*
            (film[:N-tb,i,j]-np.mean(film[:N-tb,i,j]))))
            for j in range(res2)] for i in range(res1)])
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
        a=lag1
        b=lag2
        c=lag3
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


def F(X):
    return (X-np.mean(X))
def CM(X,n=1):
    if n!=1:
        d=1
        for i in range (n):
            d=d*(X[i]-np.mean(X[i]))
            return np.mean(d)


def autocumulant_5(film, lag1=0, lag2=0, lag3=0,lag4=0):
    """Creates fourth autocumulant signal image for a set of frames.

    Args:
        film: 1D array of frames from make_film
        res: int, resolution of image

    Returns:
        SOFI image: 2D array of autocumulant values
    """
    
    N, res, res2 = np.shape(film)
    if lag1==0 and lag2==0 and lag3==0 and lag4==0:
        #d_C4 = [[(np.mean((film[:,i,j] - np.mean(film[:,i,j]))**4) - 3 * np.var(film[:,i,j])) for j in range(res)] for i in range(res)]
        d_C5 = [[(CM([film[:,i,j]]*5,n=5) - 10 * np.var(film[:,i,j])*CM([film[:,i,j]]*3,n=3)) for j in range(res)] for i in range(res)]
    if lag1*lag2*lag3*lag4!=0:
        d_C5 = np.zeros((res,res))
        #for a in range(1, lag1+1):
         #   for b in range(1, lag2+1):
          #      for c in range(1, lag3+1):
           #         for d in range(1, lag4+1):
        a=lag1
        b=lag2
        c=lag3
        d=lag4
        u=np.sort([a,b,c,d])
        t1=u[0]
        t2=u[1]
        t3=u[2]
        t4=u[3]
        im = [[CM([film[t4:,i,j],film[t4-t3:N-t3,i,j],film[t4-t2:N-t2,i,j],film[t4-t1:N-t1,i,j],film[:N-t4,i,j]],n=5)
            - np.cov(film[t4:,i,j],film[t4-t3:N-t3,i,j])[0,1]*CM([film[t4-t2:N-t2,i,j],film[t4-t1:N-t1,i,j],film[:N-t4,i,j]],n=3)
            - np.cov(film[t4:,i,j],film[t4-t2:N-t2,i,j])[0,1]*CM([film[t4-t3:N-t3,i,j],film[t4-t1:N-t1,i,j],film[:N-t4,i,j]],n=3)
            - np.cov(film[t4:,i,j],film[t4-t1:N-t1,i,j])[0,1]*CM([film[t4-t2:N-t2,i,j],film[t4-t3:N-t3,i,j],film[:N-t4,i,j]],n=3)
            - np.cov(film[t4:,i,j],film[:N-t4,i,j])[0,1]*CM([film[t4-t2:N-t2,i,j],film[t4-t1:N-t1,i,j],film[t4-t3:N-t3,i,j]],n=3)
            - np.cov(film[t4-t2:N-t2,i,j],film[t4-t3:N-t3,i,j])[0,1]*CM([film[t4:,i,j],film[t4-t1:N-t1,i,j],film[:N-t4,i,j]],n=3)
            - np.cov(film[t4-t1:N-t1,i,j],film[t4-t3:N-t3,i,j])[0,1]*CM([film[t4:,i,j],film[t4-t2:N-t2,i,j],film[:N-t4,i,j]],n=3)
            - np.cov(film[:N-t4,i,j],film[t4-t3:N-t3,i,j])[0,1]*CM([film[t4:,i,j],film[t4-t2:N-t2,i,j],film[t4-t1:N-t1,i,j]],n=3)
            - np.cov(film[t4-t1:N-t1,i,j],film[t4-t2:N-t2,i,j])[0,1]*CM([film[t4-t3:N-t3,i,j],film[t4:,i,j],film[:N-t4,i,j]],n=3)
            - np.cov(film[:N-t4,i,j],film[t4-t2:N-t2,i,j])[0,1]*CM([film[t4-t3:N-t3,i,j],film[t4:,i,j],film[t4-t1:N-t1,i,j]],n=3)
            - np.cov(film[:N-t4,i,j],film[t4-t1:N-t1,i,j])[0,1]*CM([film[t4-t2:N-t2,i,j],film[t4-t3:N-t3,i,j],film[t4:,i,j]],n=3) for j in range(res)] for i in range(res)]
        d_C5+=im
        # Cumulants may be negative, depending on fluctuation
        # pattern, so images need to be shown for absolute value.

    return d_C5



def autocumulant_6(film, lag1=0, lag2=0, lag3=0,lag4=0,lag5=0):
    """Creates fourth autocumulant signal image for a set of frames.

    Args:
        film: 1D array of frames from make_film
        res: int, resolution of image

    Returns:
        SOFI image: 2D array of autocumulant values
    """
    
    N, res, res2 = np.shape(film)
    if lag1==0 and lag2==0 and lag3==0 and lag4==0 and lag5==0:
        #d_C4 = [[(np.mean((film[:,i,j] - np.mean(film[:,i,j]))**4) - 3 * np.var(film[:,i,j])) for j in range(res)] for i in range(res)]
        d_C5 = [[(CM([film[:,i,j]]*6,n=6) - 15 * np.var(film[:,i,j])*CM([film[:,i,j]]*4,n=4)-10*CM([film[:,i,j]]*3,n=3)**2+30*CM([film[:,i,j]]*2,n=2)**3) for j in range(res)] for i in range(res)]
    if lag1*lag2*lag3*lag4!=0:
        d_C5 = np.zeros((res,res))
        #for a in range(1, lag1+1):
         #   for b in range(1, lag2+1):
          #      for c in range(1, lag3+1):
           #         for d in range(1, lag4+1):
        a=lag1
        b=lag2
        c=lag3
        d=lag4
        e=lag5
        u=np.sort([a,b,c,d,e])
        t1=u[0]
        t2=u[1]
        t3=u[2]
        t4=u[3]
        t5=u[4]
        im = [[CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t3:N-t3,i,j],film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=6)
            - np.cov(film[t5:,i,j],film[t5-t4:N-t4,i,j])[0,1]*CM([film[t5-t3:N-t3,i,j],film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=4)
            - np.cov(film[t5:,i,j],film[t5-t3:N-t3,i,j])[0,1]*CM([film[t5-t4:N-t4,i,j],film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=4)   
            - np.cov(film[t5:,i,j],film[t5-t2:N-t2,i,j])[0,1]*CM([film[t5-t3:N-t3,i,j],film[t5-t4:N-t4,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=4)
            - np.cov(film[t5:,i,j],film[t5-t1:N-t1,i,j])[0,1]*CM([film[t5-t3:N-t3,i,j],film[t5-t2:N-t2,i,j],film[t5-t4:N-t4,i,j],film[:N-t5,i,j]],n=4)
            - np.cov(film[t5:,i,j],film[:N-t5,i,j])[0,1]*CM([film[t5-t3:N-t3,i,j],film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[t5-t4:N-t4,i,j]],n=4)
            - np.cov(film[t5-t4:N-t4,i,j],film[t5-t3:N-t3,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=4)
            - np.cov(film[t5-t4:N-t4,i,j],film[t5-t2:N-t2,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t3:N-t3,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=4)               
            - np.cov(film[t5-t4:N-t4,i,j],film[t5-t1:N-t1,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t2:N-t2,i,j],film[t5-t3:N-t3,i,j],film[:N-t5,i,j]],n=4)
            - np.cov(film[t5-t4:N-t4,i,j],film[:N-t5,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[t5-t3:N-t3,i,j]],n=4)
            - np.cov(film[t5-t3:N-t3,i,j],film[t5-t2:N-t2,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=4)
            - np.cov(film[t5-t3:N-t3,i,j],film[t5-t1:N-t1,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t2:N-t2,i,j],film[:N-t5,i,j]],n=4)
            - np.cov(film[t5-t3:N-t3,i,j],film[:N-t5,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t1:N-t1,i,j],film[t5-t2:N-t2,i,j]],n=4)
            - np.cov(film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t3:N-t3,i,j],film[:N-t5,i,j]],n=4)
            - np.cov(film[t5-t2:N-t2,i,j],film[:N-t5,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t3:N-t3,i,j],film[t5-t1:N-t1,i,j]],n=4)
            - np.cov(film[t5-t1:N-t1,i,j],film[:N-t5,i,j])[0,1]*CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t3:N-t3,i,j],film[t5-t2:N-t2,i,j]],n=4)
            - CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t3:N-t3,i,j]],n=3)*CM([film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=3)
            - CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t2:N-t2,i,j]],n=3)*CM([film[t5-t3:N-t3,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=3)   
            - CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[t5-t1:N-t1,i,j]],n=3)*CM([film[t5-t2:N-t2,i,j],film[t5-t3:N-t3,i,j],film[:N-t5,i,j]],n=3)   
            - CM([film[t5:,i,j],film[t5-t4:N-t4,i,j],film[:N-t5,i,j]],n=3)*CM([film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[t5-t3:N-t3,i,j]],n=3)
            - CM([film[t5:,i,j],film[t5-t2:N-t2,i,j],film[t5-t3:N-t3,i,j]],n=3)*CM([film[t5-t4:N-t4,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=3)
            - CM([film[t5:,i,j],film[t5-t1:N-t1,i,j],film[t5-t3:N-t3,i,j]],n=3)*CM([film[t5-t2:N-t2,i,j],film[t5-t4:N-t4,i,j],film[:N-t5,i,j]],n=3)
            - CM([film[t5:,i,j],film[:N-t5,i,j],film[t5-t3:N-t3,i,j]],n=3)*CM([film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[t5-t4:N-t4,i,j]],n=3)
            - CM([film[t5-t2:N-t2,i,j],film[t5-t4:N-t4,i,j],film[t5-t3:N-t3,i,j]],n=3)*CM([film[t5:,i,j],film[t5-t1:N-t1,i,j],film[:N-t5,i,j]],n=3)
            - CM([film[t5-t1:N-t1,i,j],film[t5-t4:N-t4,i,j],film[t5-t3:N-t3,i,j]],n=3)*CM([film[t5-t2:N-t2,i,j],film[t5:,i,j],film[:N-t5,i,j]],n=3)
            - CM([film[:N-t5,i,j],film[t5-t4:N-t4,i,j],film[t5-t3:N-t3,i,j]],n=3)*CM([film[t5-t2:N-t2,i,j],film[t5-t1:N-t1,i,j],film[t5:,i,j]],n=3)
            + (np.cov(film[t5:,i,j],film[t5-t4:N-t4,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[t5-t2:N-t2,i,j])[0,1]*np.cov(film[t5-t1:N-t1,i,j],film[:N-t5,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t4:N-t4,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[t5-t1:N-t1,i,j])[0,1]*np.cov(film[t5-t2:N-t2,i,j],film[:N-t5,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t4:N-t4,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[:N-t5,i,j])[0,1]*np.cov(film[t5-t1:N-t1,i,j],film[t5-t2:N-t2,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t3:N-t3,i,j])[0,1]*np.cov(film[t5-t4:N-t4,i,j],film[t5-t2:N-t2,i,j])[0,1]*np.cov(film[t5-t1:N-t1,i,j],film[:N-t5,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t3:N-t3,i,j])[0,1]*np.cov(film[t5-t4:N-t4,i,j],film[t5-t1:N-t1,i,j])[0,1]*np.cov(film[t5-t2:N-t2,i,j],film[:N-t5,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t3:N-t3,i,j])[0,1]*np.cov(film[t5-t4:N-t4,i,j],film[:N-t5,i,j])[0,1]*np.cov(film[t5-t1:N-t1,i,j],film[t5-t2:N-t2,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t2:N-t2,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[t5-t4:N-t4,i,j])[0,1]*np.cov(film[t5-t1:N-t1,i,j],film[:N-t5,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t2:N-t2,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[t5-t1:N-t1,i,j])[0,1]*np.cov(film[t5-t4:N-t4,i,j],film[:N-t5,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t2:N-t2,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[:N-t5,i,j])[0,1]*np.cov(film[t5-t1:N-t1,i,j],film[t5-t4:N-t4,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t1:N-t1,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[t5-t2:N-t2,i,j])[0,1]*np.cov(film[t5-t4:N-t4,i,j],film[:N-t5,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t1:N-t1,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[t5-t4:N-t4,i,j])[0,1]*np.cov(film[t5-t2:N-t2,i,j],film[:N-t5,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[t5-t1:N-t1,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[:N-t5,i,j])[0,1]*np.cov(film[t5-t4:N-t4,i,j],film[t5-t2:N-t2,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[:N-t5,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[t5-t2:N-t2,i,j])[0,1]*np.cov(film[t5-t1:N-t1,i,j],film[t5-t4:N-t4,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[:N-t5,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[t5-t1:N-t1,i,j])[0,1]*np.cov(film[t5-t2:N-t2,i,j],film[t5-t4:N-t4,i,j])[0,1] 
            + np.cov(film[t5:,i,j],film[:N-t5,i,j])[0,1]*np.cov(film[t5-t3:N-t3,i,j],film[t5-t4:N-t4,i,j])[0,1]*np.cov(film[t5-t1:N-t1,i,j],film[t5-t2:N-t2,i,j])[0,1])*2 for j in range(res)] for i in range(res)]
        d_C5+=im
        # Cumulants may be negative, depending on fluctuation
        # pattern, so images need to be shown for absolute value.

    return d_C5

import cv2
import time
import numpy as np


def get_transition_matrix(vx, vy, x, y, mask):
    vx, vy = vy, vx #to change swap them as it is needed for proper calculation

    det = mask.shape[0]
    zeros_to_concat = np.zeros(
        (det, det , det , 1, 2), dtype=np.float64)

    vvx = (vx[:, np.newaxis, np.newaxis] + vx[np.newaxis,
                                              :, np.newaxis] + vx[np.newaxis, np.newaxis, :]) / 3

    vvx = vvx + x

    vvy = (vy[:, np.newaxis, np.newaxis] + vy[np.newaxis,
                                              :, np.newaxis] + vy[np.newaxis, np.newaxis, :]) / 3
    vvy = vvy + y

    vvx = vvx[mask].reshape(det, det , det )
    vvy = vvy[mask].reshape(det, det , det )

    vvx = vvx[:, :, :, np.newaxis, np.newaxis]
    vvy = vvy[:, :, :, np.newaxis, np.newaxis]

    vvvx = np.concatenate((zeros_to_concat, vvx), axis=4)
    vvvy = np.concatenate((zeros_to_concat, vvy), axis=4)

    vvvx[:, :, :, :, 0] = 1
    vvvy[:, :, :, :, 1] = 1

    # this is a translation matrix
    return np.concatenate((vvvx, vvvy), axis=3)


def get_image(d, translation_matrix, w, h):
    t1=time.time()
    im = np.sum([[[cv2.warpAffine(d[i, j, k], translation_matrix[i, j, k], (w, h)) for i in range(
                d.shape[0])] for j in range(d.shape[1])] for k in range(d.shape[2])], axis=(0, 1, 2))
    t2=time.time()
    #print(f'Img calculation took {t2-t1} seconds\n')
    return im


def get_xc3(film, vx, vy, lag1=0, lag2=0):
    det = film.shape[3] - 2  # number of detectors
    w, h = film.shape[2], film.shape[1]  # shape im img
    N = film.shape[0]  # Number of frames
    im_t = np.zeros((h, w))  # our image (in future)

    mask = np.ones((det, det, det), dtype=bool)
    
    zeros_to_concat = np.zeros(
        (det, det , det , 1, 2), dtype=np.float64)

    if lag1 == 0 and lag2 == 0:

        for x in range(h):  # loop over x positions
            for y in range(w):  # loop over y positions
                #print(f'Calculation for x={x} and y={y}')
                t1 = time.time()
                data = film[:, x, y, :-2]
                data = data - np.mean(data, axis=0)
                d = data[:, :, np.newaxis, np.newaxis] * data[:, np.newaxis, np.newaxis, :] \
                    * data[:, np.newaxis, :, np.newaxis]
                d = d.mean(axis=0)
                d = d[mask]
                d = d.reshape(det, det, det)

                translation_matrix = get_transition_matrix(
                    vx, vy, x, y, mask)
                im_t = im_t + get_image(d, translation_matrix, w, h)
                t2 = time.time()
               # print(f'took {t2-t1} seconds\n')

    if lag1 * lag2 != 0:
        tb = np.max((lag1, lag2))
        ts = np.min((lag1, lag2))
        for x in range(h):  # loop over x positions
            for y in range(w):  # loop over y positions
                #print(
                  #  f'Calculation for x={x}, y={y}, lag1={lag1} and lag2={lag2}')
                data = film[:, x, y, :-2]
                data = data - np.mean(data, axis=0)
                t1_lag = time.time()
                d = data[tb:, :, np.newaxis, np.newaxis] * data[tb - ts:N - ts, np.newaxis, np.newaxis, :] \
                    * data[:N - tb, np.newaxis, :, np.newaxis]
                d = d.mean(axis=0)
                d = d[mask]
                d = d.reshape(det, det , det )
                translation_matrix = get_transition_matrix(
                    vx, vy, x, y, mask)
                im_t = im_t + \
                    get_image(d, translation_matrix, w, h)
                t2_lag = time.time()
                #print(f'took {t2_lag - t1_lag} seconds\n')
    return im_t
          
def get_transition_matrix_2(vx, vy, x, y, mask):
    vx, vy = vy, vx #to change swap them as it is needed for proper calculation

    det = mask.shape[0]
    zeros_to_concat = np.zeros(
        (det, det , 1, 2), dtype=np.float64)

    vvx = (vx[:, np.newaxis] + vx[np.newaxis,:] ) / 2

    vvx = vvx + x

    vvy = (vy[:, np.newaxis, ] + vy[np.newaxis,:]) / 2
    vvy = vvy + y

    vvx = vvx[mask].reshape(det, det )
    vvy = vvy[mask].reshape(det, det )

    vvx = vvx[:, :, np.newaxis, np.newaxis]
    vvy = vvy[:, :, np.newaxis, np.newaxis]

    vvvx = np.concatenate((zeros_to_concat, vvx), axis=3)
    vvvy = np.concatenate((zeros_to_concat, vvy), axis=3)

    vvvx[:, :, :, 0] = 1
    vvvy[:, :, :, 1] = 1

    # this is a translation matrix
    return np.concatenate((vvvx, vvvy), axis=2)


def get_image_2(d, translation_matrix, w, h):
    #t1=time.time()
    im = np.sum([[cv2.warpAffine(d[i, j], translation_matrix[i, j], (w, h)) for i in range(
                d.shape[0])] for j in range(d.shape[1])] , axis=(0, 1))
    #t2=time.time()
    #print(f'Img calculation took {t2-t1} seconds\n')
    return im


def get_xc2(film, vx, vy, lag1=0):
    det = film.shape[3] - 2  # number of detectors
    w, h = film.shape[2], film.shape[1]  # shape im img
    N = film.shape[0]  # Number of frames
    im_t = np.zeros((h, w))  # our image (in future)

    mask = np.ones((det, det), dtype=bool)
 
    zeros_to_concat = np.zeros(
        (det, det, 1, 2), dtype=np.float64)

    if lag1 == 0:

        for x in range(h):  # loop over x positions
            for y in range(w):  # loop over y positions
                #print(f'Calculation for x={x} and y={y}')
                #t1 = time.time()
                data = film[:, x, y, :-2]
                data = data - np.mean(data, axis=0)
                d = data[:, :, np.newaxis] * data[:, np.newaxis, :] 
                d = d.mean(axis=0)
                d = d[mask]
                d = d.reshape(det, det)

                translation_matrix = get_transition_matrix_2(
                    vx, vy, x, y, mask)
                im_t = im_t + get_image_2(d, translation_matrix, w, h)
                #t2 = time.time()
                #print(f'took {t2-t1} seconds\n')

    if lag1!= 0:
        for x in range(h):  # loop over x positions
            for y in range(w):  # loop over y positions
               # print(
                #    f'Calculation for x={x}, y={y}, lag1={lag1} and lag2={lag2}')
                data = film[:, x, y, :-2]
                data = data - np.mean(data, axis=0)
                #t1_lag = time.time()
                d = data[lag1:, :, np.newaxis] * data[:N - lag1, np.newaxis, :]
                    
                d = d.mean(axis=0)
                d = d[mask]
                d = d.reshape(det, det )
                translation_matrix = get_transition_matrix_2(vx, vy, x, y, mask)
                im_t = im_t + \
                    get_image_2(d, translation_matrix, w, h)
               # t2_lag = time.time()
                #print(f'took {t2_lag - t1_lag} seconds\n')
    return im_t

                

def get_transition_matrix_4(vx, vy, x, y, mask):
    vx, vy = vy, vx  # to swap them as it is needed for a proper calculation

    det = mask.shape[0]
    zeros_to_concat = np.zeros(
        (det, det , det , det , 1, 2), dtype=np.float64)

    vvx = (vx[:, np.newaxis, np.newaxis, np.newaxis] + vx[np.newaxis, :, np.newaxis, np.newaxis]
           + vx[np.newaxis, np.newaxis, :, np.newaxis] + vx[np.newaxis, np.newaxis, np.newaxis, :]) / 4

    vvx = vvx + x

    vvy = (vy[:, np.newaxis, np.newaxis, np.newaxis] + vy[np.newaxis, :, np.newaxis, np.newaxis]
           + vy[np.newaxis, np.newaxis, :, np.newaxis] + vy[np.newaxis, np.newaxis, np.newaxis, :]) / 4

    vvy = vvy + y

    vvx = vvx[mask].reshape(det, det , det , det )
    vvy = vvy[mask].reshape(det, det , det , det )

    vvx = vvx[:, :, :, :, np.newaxis, np.newaxis]
    vvy = vvy[:, :, :, :, np.newaxis, np.newaxis]

    vvvx = np.concatenate((zeros_to_concat, vvx), axis=5)
    vvvy = np.concatenate((zeros_to_concat, vvy), axis=5)

    vvvx[:, :, :, :, :, 0] = 1
    vvvy[:, :, :, :, :, 1] = 1

    # this is a translation matrix
    return np.concatenate((vvvx, vvvy), axis=4)


def get_image_4(d, translation_matrix, w, h):
   # t1 = time.time()
    im = np.sum([[[[cv2.warpAffine(d[i, j, k, l], translation_matrix[i, j, k, l], (w, h)) for i in range(
                d.shape[0])] for j in range(d.shape[1])] for k in range(d.shape[2])] for l in range(d.shape[3])], axis=(0, 1, 2, 3))
    #t2 = time.time()
    #print(f'Img calculation took {t2-t1} seconds\n')
    return im


def get_xc4(film,vx,vy, lag1=0, lag2=0, lag3=0):
    det = film.shape[3] - 2  # number of detectors
    w, h = film.shape[2], film.shape[1]  # shape im img
    N = film.shape[0]  # Number of frames
    im_t = np.zeros((h, w))  # our image (in future)

    mask = np.ones((det, det, det, det), dtype=bool)
    
    zeros_to_concat = np.zeros(
        (det, det , det , det , 1, 2), dtype=np.float64)

    if lag1 == 0 and lag2 == 0 and lag3 == 0:

        for x in range(h):  # loop over x positions
            for y in range(w):  # loop over y positions
                print(f'Calculation for x={x} and y={y}')
                #t1 = time.time()
                data = film[:, x, y, :-2]
                data = data - np.mean(data, axis=0)
                d4 = data[:, :, np.newaxis, np.newaxis, np.newaxis] * data[:, np.newaxis, np.newaxis, :, np.newaxis] \
                    * data[:, np.newaxis, :, np.newaxis, np.newaxis] * data[:, np.newaxis, np.newaxis, np.newaxis, :]
                d4 = d4.mean(axis=0)
                d2 = data[:, :, np.newaxis] * data[:, np.newaxis, :]
                d2 = d2.mean(axis=0)
                d2 = d2**2

                d = d4 - 3 * d2
                d = d[mask]
                d = d.reshape(det, det , det , det )

                translation_matrix = get_transition_matrix_4(
                    vx, vy, x, y, mask)
                im_t = im_t + get_image_4(d, translation_matrix, w, h)
                #t2 = time.time()
                print(f'took {t2-t1} seconds\n')

    if lag1 * lag2 * lag3 != 0:
        tb = np.max((lag1, lag2, lag3))
        ts = np.min((lag1, lag2, lag3))
        tt = int(lag1 * lag2 * lag3 / (tb * ts))

        for x in range(h):  # loop over x positions
            for y in range(w):  # loop over y positions
                #print(
                    #f'Calculation for x={x}, y={y}, lag1={lag1}, lag2={lag2} and lag3={lag3}')
                #t1_lag = time.time()
                data = film[:, x, y, :-2]
                data = data - np.mean(data, axis=0)

                d4 = data[tb:, :, np.newaxis, np.newaxis, np.newaxis] * data[tb - ts:N - ts, np.newaxis, np.newaxis, :, np.newaxis] \
                    * data[tb - tt:N - tt, np.newaxis, :, np.newaxis, np.newaxis] * data[:N - tb, np.newaxis, np.newaxis, np.newaxis, :]
                d4 = d4.mean(axis=0)

                d2_1 = data[tb:, :, np.newaxis] * \
                    data[tb - ts:N - ts, np.newaxis, :]
                d2_1 = d2_1.mean(axis=0)

                d2_2 = data[tb:, :, np.newaxis] * \
                    data[tb - tt:N - tt, np.newaxis, :]
                d2_2 = d2_2.mean(axis=0)

                d2_3 = data[tb:, :, np.newaxis] * \
                    data[:N - tb, np.newaxis, :]
                d2_3 = d2_3.mean(axis=0)
                
                d2_4 = data[tb - ts:N - ts, :, np.newaxis] * \
                    data[tb - tt:N - tt, np.newaxis, :]
                d2_4 = d2_4.mean(axis=0)
                
                d2_5 = data[tb - ts:N - ts, :, np.newaxis] * \
                    data[:N - tb, np.newaxis, :]
                d2_5 = d2_5.mean(axis=0)
                
                d2_6 = data[tb - tt:N - tt, :, np.newaxis] * \
                    data[:N - tb, np.newaxis, :]
                d2_6 = d2_6.mean(axis=0)
                
                d2_ab = d2_1[:,:,np.newaxis,np.newaxis] * d2_6[np.newaxis,np.newaxis,:,:]
                d2_bc = d2_2[:,:,np.newaxis,np.newaxis] * d2_5[np.newaxis,np.newaxis,:,:]
                d2_ca = d2_3[:,:,np.newaxis,np.newaxis] * d2_4[np.newaxis,np.newaxis,:,:]

                d = d4 - d2_ab - d2_bc - d2_ca
                #d=d.mean(axis=0)
                d = d[mask]
                d = d.reshape(det, det , det , det )
                translatAion_matrix = get_transition_matrix_4(
                    vx, vy, x, y, mask)
                im_t = im_t + \
                    get_image_4(d, translation_matrix, w, h)
                #t2_lag = time.time()
               # print(f'took {t2_lag - t1_lag} seconds\n')
    return im_t


                
def find_shift_vectors(scans,PSF,itt=10,power=0.5):

    I=[]
    for i in range(scans.shape[-1]):
        #print(i)
        a=scans.sum(axis=(0,1))[:,:,i]
        I.append(a)
    I=np.array(I)
    I_f=np.fft.fft2(I)
    I_ff=np.fft.fftshift(I_f, axes=(1,2))
    S=PSF
    GF=np.zeros((scans.shape[2],scans.shape[3]))
    GF[:np.shape(S)[0],:np.shape(S)[1]]=S
    Gf=np.fft.fft2(GF)
    Imag=[]

    A=I_f[0,:,:]
    #A=np.fft.fft2(Aa)
    for k in range(itt):
        for i in range(I_ff.shape[0]-2):
            #print(i)
            B=I_f[i,:,:]
            Imag.append(np.abs(np.fft.ifft2(B/(A/(Gf)))))
        vxxx=np.zeros((scans.shape[-1]-2))
        vyyy=np.zeros((scans.shape[-1]-2))
        for i in range(scans.shape[-1]-2):
            #print(i)
            h=(Imag[i])**power
            u=find_max_index(h)
            res=photon_finder_gaussians(h, u)
            vxxx[i]=(res['xo'][0])
            vyyy[i]=(res['yo'][0])
        vxx=vxxx-vxxx.mean()
        vyy=vyyy-vyyy.mean()
        Aa=ac1_image(scans[:,:,:,:,:],-vyy,-vxx)
        A=np.fft.fft2(Aa)
    vx=-vyy
    vy=-vxx
    
    return vx, vy

def find_PSF(scans,TH=0.045):
    I=scans.sum(axis=(0,1))
    I=I.transpose(2,0,1)
    I_f=np.fft.fft2(I)
    I_ff=np.fft.fftshift(I_f, axes=(1,2))
    ll=0
    for i in range(I_ff.shape[0]-2):
        #print(i)
        B=I_f[i,:,:]
        M=np.zeros((B.shape[0],B.shape[1]))
        for i in range (B.shape[0]):
            for j in range(B.shape[1]):
                if np.abs(B[i,j])>TH*B.max():
                    M[i,j]=1
        plt.imshow(np.fft.fftshift(M))
        plt.show()
        l=M.sum()
        ll+=l
        
    kmax=np.sqrt((ll/(I_ff.shape[0]-2))/np.pi)
    d=np.shape(scans)[2]
    d2=np.shape(scans)[3]
    sss=2*d*0.225/kmax
    G2=np.zeros((d,d2))

    for x in range(d):
        for y in range(d2):
            G2[x,y]=np.exp(-((x-d//2)**2+(y-d2//2)**2)/(2*sss**2))
    return G2

def find_max_index(img):
        m=np.max(img)
        for i in range(np.shape(img)[0]):
            for j in range (np.shape(img)[1]):
                if img[i,j]==m:
                    return [(float(i), float(j))]
                
                
def H0(S,dy,kmax,E):
    dk=1/(np.shape(S)[1]*dy)
    k_c=1.3833333333333333333
    K_c=k_c/dk
    kmax=kmax*K_c
    mask=np.zeros((S.shape[0],S.shape[1]))
    kE=np.zeros((S.shape[0],S.shape[1]))
    for x in range(kE.shape[0]):
        for y in range(kE.shape[1]):
            k=np.sqrt((x-kE.shape[0]//2)**2+(y-kE.shape[1]//2)**2)
            if k<kmax:
                kE[x,y]=E*k/kmax
                mask[x,y]=1
    return kE, mask


def FR_I(imm, bf_name='Bead_FB.npy',order=1, E=0.35, kmax=8.,dx=0.05,U=0,dx_bead=0.05):
    k=1/(imm.shape[0]*dx)
    l=int(np.round(1/(k*dx_bead)))
    S=np.load(bf_name)
    S=S**order
    Ss=np.zeros((l,l))
    Ss[:np.shape(S)[0],:np.shape(S)[1]]=S
    AS=np.abs(np.fft.fftshift(np.fft.fft2(Ss)))
    AS=AS/np.max(AS)
    ll=imm.shape[0]
    if l > ll:
        AS=AS[round(l/2-ll/2):round(l/2-ll/2)+ll,round(l/2-ll/2):round(l/2-ll/2)+ll]
    else:
        Ab=np.zeros((ll,ll))
        Ab[round(ll/2-l/2):round(l/2+ll/2),round(ll/2-l/2):round(l/2+ll/2)]=AS
        AS=Ab
    H00, mask =H0(AS,dx,kmax,E)
    H1=mask*((AS+H00+U)**(-1))
    i_imm=np.fft.fftshift(np.fft.fft2(imm))
    i_imm=i_imm/np.abs(i_imm).max()
    fr_imm=np.abs(np.fft.ifft2(np.fft.ifftshift(i_imm*H1)))
    return fr_imm
        
              
def fft_interpolate2(imm, dy, m=1, kmax=8):

    H00, mask =H0(imm,dy,kmax,E=1)
    H00=0
    
    i_imm=np.fft.fftshift(np.fft.fft2(imm))
    i_imm=mask*i_imm/np.abs(i_imm).max()
    A=np.shape(imm)[0]
    B=np.shape(imm)[1]
    int_i_imm=np.zeros((m*A,m*B),dtype=complex)
    int_i_imm[round((m*A)/2-A/2):round((m*A)/2-A/2)+A,round(m*B/2-B/2):round(m*B/2-B/2)+B]=i_imm
    
    int_imm=np.abs(np.fft.ifft2(np.fft.ifftshift(int_i_imm)))
    return int_imm    
def fft_interpolate(image, m=10, mask_name=0):

    fft_image=np.fft.fftshift(np.fft.fft2(image))
    if mask_name!=0:
        mask=np.load('masks\\mask_fft_'+str(mask_name) +'.npy')
        for i in range(fft_image.shape[0]):
            for j in range(fft_image.shape[0]):
                if mask[i][j]:
                    fft_image[i][j]=0   
    fft_image_inter=np.concatenate((np.zeros((int(((fft_image.shape[0]-1)*m-fft_image.shape[0]+1)/2),(fft_image.shape[0]-1)*m+1),dtype='complex128'),np.concatenate((np.zeros((fft_image.shape[0],int(((fft_image.shape[0]-1)*m-fft_image.shape[0]+1)/2)),dtype='complex128'),fft_image,np.zeros((fft_image.shape[0],int(((fft_image.shape[0]-1)*m-fft_image.shape[0]+1)/2)),dtype='complex128')),axis=1),np.zeros((int(((fft_image.shape[0]-1)*m-fft_image.shape[0]+1)/2),(fft_image.shape[0]-1)*m+1),dtype='complex128')),axis=0)
    image_inter=np.fft.ifft2(np.fft.ifftshift(fft_image_inter))
    return image_inter

def cross_section(data, A, B):
    """Creates cross section of image from chosen point A to B.
    """
    Ax, Ay = A
    Bx, By = B
    d = np.sqrt((Ax-Bx)**2 + (Ay-By)**2)
    ABx = np.round(np.linspace(Ax,Bx, int(d))).astype(int)
    ABy = np.round(np.linspace(Ay,By, int(d))).astype(int)
    profile = [ABx, ABy]
    xsec = data[profile]
    
    return xsec
def compute_corelation2(scans,x,y,maxlag=2000):
    signals=scans[0,:,x,y,:]-scans[0,:,x,y,:].mean(axis=0)
    corelation=np.zeros(maxlag)
    N=scans.shape[1]
    for i in range(maxlag):
        for det1 in range (scans.shape[-1]):
            for det2 in range (scans.shape[-1]):
                corelation[i]+=(signals[i:N,det1]*signals[0:N-i,det2]).mean()
    return corelation

def photon_finder_gaussians(image, centers):
    
    #commented for circle=1
    
    param_names = ['angle', 'd_angle', 'offset', 'd_offset', 'amplitude', 'd_amplitude', 'xo', 'd_xo', 
                   'yo', 'd_yo', 'sx', 'd_sx', 'sy', 'd_sy']
    res = {name: [] for name in param_names}
    
    for c in centers:
        xo, yo = tuple(map(int, c))
        fit, cov_p, infodict, errmsg = sofi.gaussfit(image, circle=0, return_all=1)
        N = np.size(image)
        n = len(fit)
        s_sq = (infodict['fvec']**2).sum()/ (N-n) # infodict already contains f(x)-y, which is needed
        cov = cov_p * s_sq
        stds = [np.sqrt(cov[i][i]) for i in range(len(fit))]
        #params = dict(zip(param_names, np.round(fit, 6)))
        params = dict(zip(param_names, np.round(fit, 7)))
        #const, amp, x, y, s, angle = fit
        const, amp, x, y, sx, sy, angle = fit
        #x +=5
        #y +=5
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
        
        
    #fits = pd.DataFrame(res)
    #positions = pd.DataFrame(centers, columns=['xo', 'yo'])
        
    return res