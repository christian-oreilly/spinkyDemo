import numpy as np


# Utilitary function to get variables necessary for training the detector, sampled randomly out of the
# signals of the reader.
def getTrainSignal(reader, nbTrainPage=30, eventName = 'spindleE1', channel = 'EEG C3-CLE'):
    stages2  = [e for e in reader.events if e.name == 'Sleep stage 2']
    events   = [e for e in reader.events if e.name == eventName]

    fs = reader.getChannelFreq(channel)
    trainSig   = []
    nbEvents = []
    pages = np.random.choice(stages2, nbTrainPage)
    epoch_length = np.min([s.duration() for s in stages2])
    nbSamples = int(epoch_length*fs)
    fs = nbSamples/epoch_length
    for s in pages:
        trainSig.append(reader.read([channel], s.startTime, s.duration())[channel].signal[0:nbSamples])              
        nbEvents.append(len([e for e in events if e.startTime >= s.startTime and  
                                                      e.startTime <= s.startTime + s.duration()]))
    trainSig = np.concatenate(trainSig)
    nbEvents = np.array(nbEvents)
    
    return trainSig, nbEvents, fs, epoch_length



def getTestSignal(reader, channel = 'EEG C3-CLE'):
    pages  = [e for e in reader.events if e.name == 'Sleep stage 2']
    fs = reader.getChannelFreq(channel)
    testSig   = []
    epoch_length = np.min([s.duration() for s in pages])
    nbSamples = int(epoch_length*fs)
    fs = nbSamples/epoch_length    
    for s in pages:
        testSig.append(reader.read([channel], s.startTime, s.duration())[channel].signal[0:nbSamples])              
    testSig = np.concatenate(testSig)
    
    return testSig, fs, epoch_length







def callMatlabFunc(mlab, funcName, inputArgs, nbOutputArg):
    closeMatlab = False
    if mlab is None:
        mlab = Matlab() #Matlab(matlab='C:/Program Files/MATLAB/R2015a/bin/matlab.exe')
        mlab.start() 
        closeMatlab = True

        
    inputStr = ""
    if len(inputArgs):
        for i, arg in enumerate(inputArgs):
            mlab.set_variable("in" + str(i), arg)
            inputStr += "in" + str(i) + ","
        inputStr = inputStr[:-1]
        
    matlabCode = ""
    if nbOutputArg == 1:
        matlabCode += "out0 = "
    elif len(outputArg) > 1:
        matlabCode += "[" 
        for i in range(nbOutputArg):
            matlabCode += "out" + str(i) + ","
        matlabCode = matlabCode[:-1]    
        matlabCode += "] = " 
        
    matlabCode += funcName + "(" + inputStr + ")"

    result = mlab.run_code(matlabCode)
    
    outArgs = (mlab.get_variable("out" + str(i)) for i in range(nbOutputArg))
    
    sucess = result["success"]
    stdout = result["content"]["stdout"]
    
    if closeMatlab :
        mlab.stop()

    if not sucess:
        raise  RuntimeError(stdout)
        
    return outArgs


def training_process(mlab, trainSig, fs, epoch_length, detection_mode, sp_thresh, nbSpindles):
    return list(callMatlabFunc(mlab, "training_process", 
                   [trainSig, fs, epoch_length, detection_mode, sp_thresh, nbSpindles], 1))[0]

def sp_thresholds_ranges(mlab, trainSig, fs, epoch_length):
    return list(callMatlabFunc(mlab, "sp_thresholds_ranges", 
                   [trainSig, fs, epoch_length], 1))[0]
		   
def kp_thresholds_ranges(mlab, trainSig, fs, epoch_length):
    return list(callMatlabFunc(mlab, "kp_thresholds_ranges", 
                   [trainSig, fs, epoch_length], 1))[0]
				   
				   
				   
def test_process(mlab, testSig, fs, epoch_length, subj_name, threshold, mode='spindles'):
    return list(callMatlabFunc(mlab, "test_process", 
                   [testSig, fs, epoch_length, subj_name, mode, threshold], 2))[0]