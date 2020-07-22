# This module is based on gaussfitter module and primarily fits 2 2d gaussians.
# User may encouter problems when fitting 1 2d gaussian as well as 2 gaussians
# that are eliptic (+ rotated) and offset and have different amplitudes, as
# no full debugging was performed. May want to use original gaussfitter for
# unsupported fits. For specific comments see NOTE from 25.04.19 in the gaussfit function.
# created by Adam Ginsburg <adam.g.ginsburg@gmail.com>
# modified by aleksandra sroda, univeristy of warsaw



from numpy import *
from scipy import optimize
from scipy import stats

def moments(data,circle,rotate,vheight):
    """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output 
    a subset of the above"""
    total = data.sum()
    X, Y = indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
    width = ( width_x + width_y ) / 2.
    height = stats.mode(data.ravel())[0][0]
    amplitude = data.max()-height
    mylist = [amplitude,x,y]
    if vheight==1:
        mylist = [height] + mylist
    if circle==0:
        mylist = mylist + [width_x,width_y]
    else:
        mylist = mylist + [width]
    if rotate==1:
        mylist = mylist + [0.] #rotation "moment" is just zero...
    return tuple(mylist)

def twodgaussian(inpars, circle, rotate, vheight):
    """Returns a 2d gaussian function of the form:
        x' = cos(rota) * x - sin(rota) * y
        y' = sin(rota) * x + cos(rota) * y
        (rota should be in degrees)
        g = b + a exp ( - ( ((x-center_x)/width_x)**2 +
        ((y-center_y)/width_y)**2 ) / 2 )

        However, the above values are passed by list.  The list should be:
        inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)

        You can choose to ignore / neglect some of the above input parameters using the following options:
            circle=0 - default is an elliptical gaussian (different x, y widths), but can reduce
                the input by one parameter if it's a circular gaussian
            rotate=1 - default allows rotation of the gaussian ellipse.  Can remove last parameter
                by setting rotate=0
            vheight=1 - default allows a variable height-above-zero, i.e. an additive constant
                for the Gaussian function.  Can remove first parameter by setting this to 0
        """
    inpars_old = inpars
    inpars = list(inpars)
    if vheight == 1:
        height = inpars.pop(0)
        height = float(height)
    else:
        height = float(0)
    amplitude, center_x, center_y = inpars.pop(0),inpars.pop(0),inpars.pop(0)
    amplitude = float(amplitude)
    center_x = float(center_x)
    center_y = float(center_y)
    if circle == 1:
        width = inpars.pop(0)
        width_x = float(width)
        width_y = float(width)
    else:
        width_x, width_y = inpars.pop(0),inpars.pop(0)
        width_x = float(width_x)
        width_y = float(width_y)
    if rotate == 1:
        rota = inpars.pop(0)
        rota = pi/180. * float(rota)
        rcen_x = center_x * cos(rota) - center_y * sin(rota)
        rcen_y = center_x * sin(rota) + center_y * cos(rota)
    else:
        rcen_x = center_x
        rcen_y = center_y
    if len(inpars) > 0:
        raise ValueError("There are still input parameters:" + str(inpars) + \
                " and you've input: " + str(inpars_old) + " circle=%d, rotate=%d, vheight=%d" % (circle,rotate,vheight) )
            
    def rotgauss(x,y):
        if rotate==1:
            xp = x * cos(rota) - y * sin(rota)
            yp = x * sin(rota) + y * cos(rota)
        else:
            xp = x
            yp = y
        g = height+amplitude*exp(
            -(((rcen_x-xp)/width_x)**2+
            ((rcen_y-yp)/width_y)**2)/2.)
        return g
    return rotgauss


def two2dgaussians(inpars, circle, rotate, vheight):
    """Returns a sum of two 2d gaussian functions of the form:
        x' = cos(rota) * x - sin(rota) * y
        y' = sin(rota) * x + cos(rota) * y
        (rota should be in degrees)
        g = b + a exp ( - ( ((x-center_x)/width_x)**2 +
        ((y-center_y)/width_y)**2 ) / 2 )

        However, the above values are passed by list.  The list should be:
        inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)x2
        the list is split in two for two gaussians

        You can choose to ignore / neglect some of the above input parameters using the following options:
            circle=0 - default is an elliptical gaussian (different x, y widths), but can reduce
                the input by one parameter if it's a circular gaussian
            rotate=1 - default allows rotation of the gaussian ellipse.  Can remove last parameter
                by setting rotate=0
            vheight=1 - default allows a variable height-above-zero, i.e. an additive constant
                for the Gaussian function.  Can remove first parameter by setting this to 0
        """
    
    
    
    inpars = list(inpars)
    sigma = inpars.pop(-1)
    sigma = float(sigma)
    amp = inpars.pop(0)
    amp = float(amp)
    length = len(inpars)
    """if length%2 == 1:
        sigma = inpars.pop(-1)
        sigma = float(sigma)
        length = len(inpars)
        if length%2 == 1:
            raise ValueError("Something is wrong, two gaussians = even number of parms [as 2019].")
        else:
            pass"""
    inpars1 = inpars[:int(length/2)]
    inpars2 = inpars[int(length/2):]
    inpars_old1 = inpars1
    inpars_old2 = inpars2
    inpars1 = list(inpars1)
    inpars2 = list(inpars2)
        #print(inpars1, inpars2)
    """if vheight == 1:
        height1 = inpars1.pop(0)
        height1 = float(height1)
        height2 = inpars2.pop(0)
        height2 = float(height2)
    else:
        height1 = float(0)
        height2 = float(0)"""
    center_x1, center_y1 = inpars1.pop(0),inpars1.pop(0)
    center_x2, center_y2 = inpars2.pop(0),inpars2.pop(0)
#     amplitude1, center_x1, center_y1 = inpars1.pop(0),inpars1.pop(0),inpars1.pop(0)
#     amplitude1 = float(amplitude1)
#     amplitude2, center_x2, center_y2 = inpars2.pop(0),inpars2.pop(0),inpars2.pop(0)
#     amplitude1 = float(amplitude2)
    amplitude1 = float(amp)
    amplitude2 = float(amp)
    center_x1 = float(center_x1)
    center_y1 = float(center_y1)
    center_x2 = float(center_x2)
    center_y2 = float(center_y2)
    if circle == 1:
        #width1 = inpars1.pop(0)
        width1 = sigma
        width_x1 = float(width1)
        width_y1 = float(width1)
        width2 = sigma
        #width2 = inpars2.pop(0)
        width_x2 = float(width2)
        width_y2 = float(width2)
    else:
        width_x1, width_y1 = inpars1.pop(0),inpars1.pop(0)
        width_x1 = float(width_x1)
        width_y1 = float(width_y1)
        width_x2, width_y2 = inpars2.pop(0),inpars2.pop(0)
        width_x2 = float(width_x2)
        width_y2 = float(width_y2)
    if rotate == 1:
        rota1 = inpars1.pop(0)
        rota1 = pi/180. * float(rota1)
        rcen_x1 = center_x1 * cos(rota1) - center_y1 * sin(rota1)
        rcen_y1 = center_x1 * sin(rota1) + center_y1 * cos(rota1)
        rota2 = inpars2.pop(0)
        rota2 = pi/180. * float(rota2)
        rcen_x2 = center_x2 * cos(rota2) - center_y2 * sin(rota2)
        rcen_y2 = center_x2 * sin(rota2) + center_y2 * cos(rota2)
    else:
        rcen_x1 = center_x1
        rcen_y1 = center_y1
        rcen_x2 = center_x2
        rcen_y2 = center_y2
    if len(inpars1) > 0:
        raise ValueError("There are still input parameters:" + str(inpars1) + \
                " and you've input: " + str(inpars_old1) + " circle=%d, rotate=%d, vheight=%d" % (circle,rotate,vheight) )
    if len(inpars2) > 0:
        raise ValueError("There are still input parameters:" + str(inpars2) + \
                " and you've input: " + str(inpars_old2) + " circle=%d, rotate=%d, vheight=%d" % (circle,rotate,vheight) )
            
            
    def tworotgauss(x,y):
        if rotate==1:
            xp1 = x * cos(rota1) - y * sin(rota1)
            yp1 = x * sin(rota1) + y * cos(rota1)
            xp2 = x * cos(rota2) - y * sin(rota2)
            yp2 = x * sin(rota2) + y * cos(rota2)
        else:
            xp1 = x
            yp1 = y
            xp2 = x
            yp2 = y
        width_y = (width_y1+width_y2)/2
        width_x = (width_x1+width_x2)/2
        g = amplitude1*exp(-(((rcen_x1-xp1)/width_x)**2+
                         ((rcen_y1-yp1)/width_y)**2)/2.)+amplitude2*exp(-(((rcen_x2-xp2)/width_x)**2+
                         ((rcen_y2-yp2)/width_y)**2)/2.)
        return g
    return tworotgauss


def gaussfit(data, err=None, params1=[],params2=[], centers=None, two=False,autoderiv=1, return_all=0, circle=0, rotate=1, vheight=1):
    """
    Gaussian fitter with the ability to fit a variety of different forms of 2-dimensional gaussian and two neighbouring 2d gaussians.
    
    Input Parameters:
        data - 2-dimensional data array
        err=None - error array with same size as data array
        params=[] - initial input parameters for Gaussian function.
            (height, amplitude, x, y, width_x, width_y, rota)
        autoderiv=1 - use the autoderiv provided in the lmder.f function (the alternative
            is to us an analytic derivative with lmdif.f: this method is less robust)
        return_all=0 - Default is to return only the Gaussian parameters.  See below for
            detail on output
        circle=0 - default is an elliptical gaussian (different x, y widths), but can reduce
            the input by one parameter if it's a circular gaussian
        rotate=1 - default allows rotation of the gaussian ellipse.  Can remove last parameter
            by setting rotate=0
        vheight=1 - default allows a variable height-above-zero, i.e. an additive constant
            for the Gaussian function.  Can remove first parameter by setting this to 0

    Output:
        Default output is a set of Gaussian parameters with the same shape as the input parameters
        Can also output the covariance matrix, 'infodict' that contains a lot more detail about
            the fit (see scipy.optimize.leastsq), and a message from leastsq telling what the exit
            status of the fitting routine was
    """
    
    
    """ 25.04.2019 AS: IMPORTANT NOTE!!!!
    1. This function was created primarily for fitting 2 2d gaussians, so 
    there might be issues when fitting only 1.
    2. Input parameters for fit (from the moments function) are created 
    for each gauss separately and are in form: (amp, x, y, sigma).
    NOTE THAT THERE IS NO OFFSET AND CIRCLE=1!!!! I dont know what happens
    if you add offset and enable elipse and rotation, i.e. if the input parms are
    passed correctly to the errorfunction.
    conversion of moments1+moments2 --> input par for errorfunction:
    (amp1, x1, y1, sigma1), (amp2, x2, y2, sigma2) --> 
    --> ((amp1+amp2)/2, x1, y1, x2, y2, (sigma1+sigma2)/2)
    """
    # checking if input parameters provided. if not, they are 
    # created by the "moments" function.
    
    if (params1 == [] and params2 == [] and not two):
        params = (moments(data,circle,rotate,vheight))
    elif (params1 != [] or params2 != [] and not two):
        params = [*params1, *params2]
    elif (params1 == [] or params2 == []) and centers==None and two:
        #params = (moments(data,circle,rotate,vheight))
        raise ValueError("I'm sorry, you must provide centers if you fit 2 gaussians and don't have input parms [as 2019].")
    elif (params1 == [] or params2 == []) and centers!=None and two:
        center1, center2 = centers
        x1, y1 = center1
        x2, y2 = center2
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        params1 = list(moments(data[x1-4:x1+4,y1-4:y1+4],circle,rotate,vheight))
        params2 = list(moments(data[x2-4:x2+4,y2-4:y2+4],circle,rotate,vheight))
        sigma = (params1[-1]+params2[-1])/2  # WORKS ONLY FOR CIRCLE=1!! 
        amp = (params1[0]+params2[0])/2
        params1[1] += x1-4
        params1[2] += y1-4
        params2[1] += x2-4
        params2[2] += y2-4
        #print(params1, params2)
        params1.pop(-1)
        params2.pop(-1)
        params1.pop(0)  # getting rid of the amplitude
        params2.pop(0)
        """
        if circle==0:
            h1, amp1, X1, Y1, sx1, sy1, fi1 = (moments(data[x1-4:x1+4,y1-4:y1+4],circle,rotate,vheight))
            h2, amp2, X2, Y2, sx2, sy2, fi2 = (moments(data[x2-4:x2+4,y2-4:y2+4],circle,rotate,vheight))
            # need to shift X,Y to fit the whole image, as right now
            # they represent centers in a cropped data[x1-4:x1+4,y1-4:y1+4] image!!!!!!
            X1 += x1-4
            Y1 += y1-4
            X2 += x2-4
            Y2 += y2-4
            params1 = h1, amp1, X1, Y1, sx1, sy1, fi1
            params2 = h2, amp2, X2, Y2, sx2, sy2, fi2
        elif circle==1:
            h1, amp1, X1, Y1, s1, fi1 = (moments(data[x1-4:x1+4,y1-4:y1+4],circle,rotate,vheight))
            h2, amp2, X2, Y2, s2, fi2 = (moments(data[x2-4:x2+4,y2-4:y2+4],circle,rotate,vheight))
            # need to shift X,Y to fit the whole image, as right now
            # they represent centers in a cropped data[x1-4:x1+4,y1-4:y1+4] image!!!!!!
            X1 += x1-4
            Y1 += y1-4
            X2 += x2-4
            Y2 += y2-4
            params1 = h1, amp1, X1, Y1, s1, fi1
            params2 = h2, amp2, X2, Y2, s2, fi2"""
        params = [*[amp], *params1, *params2, *[sigma]]  
        #print("1:", params1, "\n2:", params2)
        #print("\n",params)
        
     
    # checking if error array is provided.
    
    if (err == None and not two):
        errorfunction = lambda p: ravel((twodgaussian(p,circle,rotate,vheight)(*indices(data.shape)) - data))
    elif err == None and two:
        errorfunction = lambda p: ravel((two2dgaussians(p,circle,rotate,vheight)(*indices(data.shape)) - data))                     
    elif (err != None and not two):
        errorfunction = lambda p: ravel((twodgaussian(p,circle,rotate,vheight)(*indices(data.shape)) - data)/err)
    else:
        errorfunction = lambda p: ravel((two2dgaussians(p,circle,rotate,vheight)(*indices(data.shape)) - data)/err)
    
    
    if autoderiv == 0:
        raise ValueError("I'm sorry, I haven't implemented this feature yet.")
    else:
        p, cov, infodict, errmsg, success = optimize.leastsq(errorfunction, params, full_output=1)                    
    if  return_all == 0:
        return p,cov
    elif return_all == 1:
        return p,cov,infodict,errmsg,success
    
    
    
    
    
# this post was helpful when cov matrix was singular and i didnt know what to do:
# https://stats.stackexchange.com/questions/70899/what-correlation-makes-a-matrix-singular-and-what-are-implications-of-singularit
# i had to get rid of redundant parameters (different amps and sigmas for both gaussians and offset) - as
