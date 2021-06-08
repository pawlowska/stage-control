import subprocess
import numpy as np

def photoncounting(int_time, file_name= r"C:\Users\PREZES\Documents\stage-control\SPAD_photocounting\results\temp", spad_file= r"C:\Users\PREZES\Documents\stage-control\SPAD_photocounting\nanoSPAD.exe"):
    '''The function runs the photon counting measurement i.e. saves the number of counts at each detector in the given time
    Args:
    int_time - The integration time of measurement (the duration of photoncounting measurement) in (10's of ns, not sure, Olek pleas check)
    file_name - The path to file where the photoncounting data will be saved as a binary file
    spad_file - The path to file with spad photoncounting program. (Rather don't change it)'''
    subprocess.run([spad_file,
                   "-t",
                   str(file_name),
                   str(0),
                   str(0),
                   str(int_time)])

def readPhotoncounts(file_name= r"C:\Users\PREZES\Documents\stage-control\SPAD_photocounting\results\temp"):
    '''The function reads the photoncounting data and returns the vector of total counts at each SPAD pixel during the measurement
    Args:
    file_name - The path to file where the photoncounting data was saved as a binary file'''
    
    d=np.fromfile(file_name,dtype =np.uint8)
    d=d.reshape(np.shape(d)[0]//4,4)
    d=np.uint64(d)
    d[:,1]=d[:,1]*(2**8)
    d[:,2]=d[:,2]*(2**16)
    d[:,3]=d[:,3]*(2**24)
    I=(d.sum(axis=1))
    return I