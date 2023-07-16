__name__ = "SignalArray"
__author__ = "Daniel Ferrer-Vinals"
__all__ = ["read_scan", "compute_params", "data_describe", "data_transform", "gapped_sec", "signal_map"]

import pandas as pd
import numpy as np

def read_scan(filename, pep_len = 20, score_res = 20):
    '''
    Read epitope scan raw data from linear peptide arrays, in csv format
    Parameters
    ----------
    filename: list
        A list of strings. Each list element stores a string with the filename 
        for the epitope scan data file.
    pep_len: int. Optional
        Number of residues of each peptide on the array. Default: 20
    score_res: int. Optional
        Map the peptide score to the N-th residue of the peptide. Default: 20 (end of peptide)
    Returns
    ----------
    df: pandas.DataFrame object that contains the peptide array scan data
    '''

    if not type(pep_len) is int: 
        raise TypeError("pep_len : only integers are allowed ")
    elif not type(score_res) is int:
         raise TypeError("score_res : only integers are allowed ")
    elif pep_len<score_res:
        raise Exception("score_res can't be higher than pep_len")
        
    elif pep_len != 20 or score_res != 20:    
        s = (score_res) - pep_len -1 
    else:
        s =-1

    df= pd.read_csv(filename)
    scor_res = df['Seq'].str[s]
    df['s_res'] = scor_res

    return df


def compute_params(dataframe, combine = 'mean', flag_noisy= True):
    
    '''
    Compute statistic descriptors on a given scan data from linear peptide arrays, in csv format.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame 
        A Pandas.DataFrame object that contains the peptide array scan data
    combine: str. Optional
        'max' or 'mean'. Defines how to combine the spot replicates from the array,
        to give the spot signal. 
        If 'max' is passed, the spot signal is determined by the higher score
        replicate. 
        If 'mean' is passed, the spot signal is determined by the geometric mean 
        between replicates if they do not deviate more than 40%, otherwise the 
        corresponding signal equals the replicate with the higest score value. 
        Default: 'mean'
    flag_noisy: bool. Optional
        if 'True', a "flag" of 0 or 1 is introduced at every peptide entry on the array: 1 if 
        the deviation between replicates is higher than 40%, otherwise 0. 
        Default: 'True'
    Returns
    ----------
    df: dataframe with the selected parameters
    '''
    df= dataframe
    #mean
    df['ave'] = df.iloc[:,[1,2]].mean(axis = 1) 
    #mean deviation
    df['avedev'] = ((df.r1 - df.ave).abs() + (df.r2 - df.ave).abs())/2
    # percent deviation between replicates
    df['dev_ratio'] = df.apply(lambda x:0 if x.avedev==0 else x.avedev/x.ave, axis=1)
    
    # signal value:
    if combine == 'max':
        df['combined_signal'] = df.apply(lambda x:max(x.r1, x.r2) if x.dev_ratio >=0.4 else x.ave, axis=1)
    elif combine == 'mean':
        df['combined_signal'] = df.apply(lambda x:x.ave if x.dev_ratio <= 0.4 else 0, axis=1)
    
    if flag_noisy:
        df['flag'] = df.apply(lambda x:0 if x.dev_ratio <= 0.4 else 1, axis=1)
    return df


def data_describe(dataframe):
    '''
    Data exploratory function that sumarizes some global descriptors of the array data. Useful to 
    stablish "cut-off lines" i.e. threshold.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame 
        A Pandas.DataFrame object that contains the peptide array scan data
    Returns
    ----------
    df: dataframe with the global descriptors  
    '''
    df = dataframe
    d = df.describe()
    B = d.iloc[2:,[0,1,2,5]]
    return B


def data_transform(dataframe, method = 'linear', threshold=0):
    
    '''
    This function implements linear or non-linear transformations on a pandas.dataframe
    object containing the array data.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame 
        A Pandas.DataFrame object that contains the peptide array scan data
    method: str. Optional
        One of: 'linear', 'sqrt', 'cubic', 'log' (Default: 'linear')
    signal_threshold: int. Optional
        Set threshold to a 'value', if a spot's signal value is < threshold, then the spot signal 
        is replaced by 0. Default: threshold = 0 (no threshold)        
    Returns
    ----------
    df: dataframe with the selected data trasnfomation method 
    '''

    df = dataframe
    t = threshold
    
    if method == 'linear':
        
        df['linear']= df.apply(lambda x: max(0, x.combined_signal-t), axis=1)
        df['signal_plot'] = df.apply(lambda x: x.linear/df['linear'].max(), axis=1)
        
    elif method == 'sqrt': 
        df['sqrt'] = df.apply(lambda x: np.sqrt(max(0, x.combined_signal-t)), axis=1)
        df['signal_plot'] = df.apply(lambda x: x.sqrt/df['sqrt'].max(), axis=1)
        
    elif method == 'cubic': 
        df['cubic'] = df.apply(lambda x: np.cbrt(max(0, x.combined_signal-t)), axis=1)
        df['signal_plot'] = df.apply(lambda x: x.cubic/df['cubic'].max(), axis=1)     
        
    elif method == 'log': 
        df['log'] = df.apply(lambda x: np.log10(max(1, x.combined_signal-t)), axis=1)
        df['signal_plot'] = df.apply(lambda x: x.log/df['log'].max(), axis=1)  
    

        
        
def gapped_seq(dataframe, seq_trace, p_len, overlap_step=1):
    
    '''
    Build a list of tuples:('aa_symbol': signal_plot). List elements match 
    the position and sequence of the symbols in the alignment, including gaps.

    The seq_trace are built from the alignment trace. Gaps are represented by 'None'.

    Parameters
    ----------
    dataframe: Dataframe (df)
        A Pandas dataframe that contains scan data and a 'score_res' column
        mapping the peptide score to the N-th residue of the peptide.  
    seq_trace: list
        A list of one-letter-code of amino acids.The seq_trace are built 
        from the alignment trace. Gaps are represented by 'None'.
    overlap_step: int. Optional
        The number of residues skipped between two consecutive overlaping peptides (Default: 1).
        example: an array of 20-mer peptides with 19 residue overlap, has overlap_step= 1. 
    p_len: int. 
        Number of residues of each peptide on the array. 
    Returns
    ----------
    gapped_seq: list of tuples
    
    See Also
    --------
    signalarray.read_scan()
    biotite.sequence.align.get_symbols()
    '''
    
    template = seq_trace
    df = dataframe
    step = overlap_step        
    gapped = list(zip(df.s_res , df.signal_plot)) 
    lk1 =  df["s_res"].values.tolist()
    plen = p_len
    
    if step == 1:
        x, b = 0, 0
        c = 0                      # cyclic counter up to the peptide length :20
        p = 0                      # peptide counter
        for b in range(len(lk1)):
            for a in template[x:]:
                if c < plen-1 : 
                    if a==None:
                        gapped.insert(x,(template[x],0)) 
                        x=x+1
                    elif a != lk1[b]:
                        gapped.insert(x,(template[x],0))         
                        x=x+1
                        c=c+1
                    elif p==0:
                        gapped.insert(x,(template[x],0)) 
                        x=x+1
                        c=c+1 
                    else:
                        x=x+1
                        c=c+1 
                        break
                else:
                    c = 0 # reset the counter        
                    p=p+1
                    x=x+1
                    break

    elif step == 2:
        x, b = 0, 0
        c=0 
        p=0 
        for b in range(len(lk1)):
            for a in template[x:]:
                if c < plen-1 and p==0:            
                    if a==None:
                        gapped.insert(x,(template[x],0)) 
                        x=x+1
                    else:
                        gapped.insert(x,(template[x],0))         
                        x=x+1
                        c=c+1
                elif p==0 :
                    c = 0 # reset the counter        
                    p=p+1
                    x=x+1
                    break
                if p!=0: 
                    if a==None and c == 0:
                        gapped.insert(x,(template[x],0)) 
                        x=x+1
                    elif c % 2 == 0: 
                        if a==None:
                            gapped.insert(x,(template[x],0)) 
                            x=x+1
                        else:
                            gapped.insert(x,(template[x],0)) 
                            x=x+1
                            c=c+1
                    elif c % 2 != 0: 
                        if a==None:
                            gapped.insert(x,(template[x],0)) 
                            x=x+1
                        elif a != lk1[b]:
                            gapped.insert(x,(template[x],0))         
                            x=x+1
                            c=c+1
                        else:        
                            x=x+1
                            c=c+1
                            break

    if len(gapped) < len(template) and template[len(gapped)+1]== None:            
        gapped_tail=[]
        for n in range(len(template)-len(gapped)):
            gapped_tail.append(('None', 0))                
        gapped = gapped + gapped_tail
   
    return gapped


def signal_map(gapped_seq1, gapped_seq2,):
    '''
    Builds a numpy.ndarray that maps peptide signal score into the corresponding 
    position of the score residue on the sequence alignment.
    
    Parameters
    ----------
    gapped_seq1: list
        A list of tuples('aa_symbol': signal_plot) for sequence[0] on the aligment.
    gapped_seq2: list
        A list of tuples('aa_symbol': signal_plot) for sequence[1] on the aligment.
    
    Returns
    ----------
    fl_score: numpy.ndarray
    
    See Also
    --------
    signalarray.gapped_seq()
    
    '''
    gapd_s1 = gapped_seq1
    gapd_s2 = gapped_seq2
    fl_score = np.zeros((len(gapd_s1),2))
    for v1 in range(len(gapd_s1)):
        fl_score[v1,0] = gapd_s1[v1][1]    
        fl_score[v1,1] = gapd_s2[v1][1]
        
    return fl_score


        
        
        
        
        
           
        




