from pymatbridge import Matlab
import pandas as pd

def readDetectorOutput(fileName, mode="spindles"):
    inPage = False
    isStart = False
    pageNo = 0
    numberSpindles = -1
    
    if mode == "spindles":    
        results = {"page":[], "start":[], "end":[]}
    elif mode == "kcomplex":
        results = {"page":[], "time":[]}
    else:
        raise ValueError("The mode argument must be equal to spindles or kcomplex.")
    
    with open(fileName, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            elif inPage:
                if mode == "spindles":
                    if isStart:
                        results["page"].append(pageNo)
                        results["start"].append(float(line))
                        isStart = False
                    else:
                        results["end"].append(float(line))
                        numberSpindles -= 1
                        if numberSpindles:
                            isStart = True
                        else:
                            inPage = False
                elif mode == "kcomplex":
                    results["page"].append(pageNo)
                    results["time"].append(float(line))
                    numberSpindles -= 1
                    if numberSpindles == 0:
                        inPage = False      
                else:
                    raise ValueError("The mode argument must be equal to spindles or kcomplex.")
            else:
                pageNo, numberSpindles = line.split(" ")
                pageNo = int(pageNo)
                numberSpindles = int(numberSpindles)
                if numberSpindles:
                    inPage = True
                    isStart = True

    return pd.DataFrame(results)


def sensitivity(TP, TN, FP, FN):  
    if TP > 0 :
        return TP/float(TP + FN)    
    else:
        return 0.0        


def specificity(TP, TN, FP, FN):  
    if TN > 0 :
        return TN/float(FP + TN)
    else:
        return 0.0 


def accuracy(TP, TN, FP, FN):  
    if TN > 0 :
        N = FN + TN +  FP + TP
        return (TN + TP)/float(N)
    else:
        return 0.0                


def PPV(TP, TN, FP, FN):    # Positive predictive value                 
    if TP > 0 :
        return TP/float(TP + FP)      
    else:
        return 0.0


def NPV(TP, TN, FP, FN):    # Negative predictive value                        
    if TN > 0 :
        return TN/float(TN + FN)
    else:
        return 0.0 


def MCC(TP, TN, FP, FN):    # Matthew's correlation coefficient                        
    if TN > 0 :
        num = float(TP*TN - FP*FN)

        # NB: Casting as integers since Python integers has no maximal 
        # value wheras numpy.int64 does. Without these casts, when analyzing
        # whole nights, the operation P*P2*N*N2 can overflow.
        P   = int(TP + FN)
        P2  = int(TP + FP)
        N   = int(FP + TN)
        N2  = int(FN + TN)
        den = (P*P2*N*N2)**0.5
        if den == 0:
            return 0.0
        return num/den
    else:
        return 0.0 


def randomAggreProb(TP, TN, FP, FN):    # Negative predictive value                        
    if TN > 0 :
        P   = TP + FN
        P2  = TP + FP
        N   = FP + TN
        N2  = FN + TN
        return float(P2*P + N2*N)/float((P+N)**2)
    else:
        return 0.0 


def cohenk(TP, TN, FP, FN):    # Negative predictive value                        
    if TN > 0 :
        Pe  = randomAggreProb(TP, TN, FP, FN)
        acc = accuracy(TP, TN, FP, FN) 
        return (acc - Pe)/(1-Pe)
    else:
        return 0.0     

def F1(TP, TN, FP, FN):    # Negative predictive value                        
    if TN > 0 :
        return (2.0*TP)/(2.0*TP + FP+FN)
    else:
        return 0.0 




def callMatlabFunc(mlab, funcName, inputArgs, nbOutputArg, debug=False, setupCode=""):

    if debug:
        print("Entering callMatlabFunc...")    

    closeMatlab = False
    if mlab is None:
        if debug:
            print("Starting Matlab...")
        mlab = Matlab() #Matlab(matlab='C:/Program Files/MATLAB/R2015a/bin/matlab.exe')
        mlab.start() 
        closeMatlab = True

    if len(setupCode):
        result = mlab.run_code(setupCode)
        if not result["success"]:
            raise  RuntimeError(result["content"]["stdout"]  )

    if debug:
        print("Setting input variables...")    
        
    inputStr = ""
    if len(inputArgs):
        for i, arg in enumerate(inputArgs):
            mlab.set_variable("in" + str(i), arg)
            inputStr += "in" + str(i) + ","
        inputStr = inputStr[:-1]
        
    if debug:
        print("Input variables set...")    
        
        
        
    matlabCode = ""
    if nbOutputArg == 1:
        matlabCode += "out0 = "
    elif nbOutputArg > 1:
        matlabCode += "[" 
        for i in range(nbOutputArg):
            matlabCode += "out" + str(i) + ","
        matlabCode = matlabCode[:-1]    
        matlabCode += "] = " 
        
    matlabCode += funcName + "(" + inputStr + ")"

    if debug:
        print("Matlab Code: ")
        print(matlabCode)
        
    result = mlab.run_code(matlabCode)

    if debug:
        print("run_code executed.")
        print(result)
    
    outArgs = [mlab.get_variable("out" + str(i)) for i in range(nbOutputArg)]

    if debug:
        print("Out args: ")
        print(outArgs)
    
    sucess = result["success"]
    stdout = result["content"]["stdout"]
    
    if closeMatlab :
        if debug:
            print("Stoping Matlab...")
        mlab.stop()

    if not sucess:
        raise  RuntimeError(stdout)
        
    return outArgs


def training_process(mlab, trainSig, fs, epoch_length, detection_mode, sp_thresh, nbEvents):
    setupCode = "train = @(trainSig, fs, epoch_length, detection_mode, sp_thresh, nbEvents) training_process(data_epoching(trainSig, epoch_length), fs, detection_mode, sp_thresh, nbEvents, 'Off');"
    return callMatlabFunc(mlab, "train", 
                   [trainSig, fs, epoch_length, detection_mode, sp_thresh, nbEvents], 1, setupCode=setupCode, debug=True)[0]

    #return callMatlabFunc(mlab, "training_process", 
    #               [trainSig, fs, epoch_length, detection_mode, sp_thresh, nbSpindles], 1)[0]



def sp_thresholds_ranges(mlab, trainSig, epoch_length, fs):    
    setupCode = "threshold_range = @(trainSig, epoch_length, fs) sp_thresholds_ranges(data_epoching(trainSig, epoch_length), fs);"
    return callMatlabFunc(mlab, "threshold_range", 
                   [trainSig, epoch_length, fs], 1, setupCode=setupCode)[0]

		   
def kp_thresholds_ranges(mlab, trainSig, epoch_length, fs):
    setupCode = "threshold_range = @(trainSig, epoch_length, fs) kp_thresholds_ranges(data_epoching(trainSig, epoch_length), fs);"
    return callMatlabFunc(mlab, "threshold_range", 
                   [trainSig, epoch_length, fs], 1, setupCode=setupCode)[0]

				   
				   
def test_process(mlab, testSig, fs, epoch_length, subj_name, threshold, mode='spindles'):
    setupCode = "test = @(testSig, fs, epoch_length, subj_name, mode, threshold) test_process(data_epoching(testSig, epoch_length), fs, subj_name, mode, threshold);"
    return callMatlabFunc(mlab, "test", 
                   [testSig, fs, epoch_length, subj_name, mode, threshold], 2, setupCode=setupCode)[0]

    #return callMatlabFunc(mlab, "test_process", 
    #               [testSig, fs, epoch_length, subj_name, mode, threshold], 2)[0]


