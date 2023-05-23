def ind_eac(inputs):
    inputs=inputs.astype('float')
    sum_dpc, aa, previous_eac = inputs.iloc[:,0], inputs.iloc[:,1], inputs.iloc[:,2]
    sum_dpc[sum_dpc>0.5]=0.5
    result = aa*sum_dpc*2+previous_eac*(1-sum_dpc*2)
    return result