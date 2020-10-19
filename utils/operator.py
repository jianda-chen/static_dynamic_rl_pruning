import torch

def convert_01_to_n1p1(x):
    '''
    convert x from zero-one to negative-one-positive-one
    x must be {0, 1}
    '''
    assert x.dtype == torch.float32 or x.dtype == torch.long
    if x.dtype == torch.float32:
        x = x * 2. - 1.
    elif x.dtype == torch.long:
        x = x * 2 - 1

    return x

def convert_n1p1_to_01(x):
    '''
    convert x from negative-one-positive-one to zero-one
    x must be {-1, 1}
    '''
    assert x.dtype == torch.float32 or x.dtype == torch.long
    if x.dtype == torch.float32:
        x = (x + 1.) / 2.
    elif x.dtype == torch.long:
        x = (x + 1) / 2

    return x

def convert_n10p1_to_01(x, zero_to_p1):
    '''
    convert x from negative-one-zeros-positive-one to zero-one
    x usually is output of torch.sign()
    x must be {-1, 0, 1}
    x should be float
    zero_to_ones is boolean
    '''
    assert x.dtype == torch.float32
    assert zero_to_p1 == True or zero_to_p1 == False
    if zero_to_p1 == True:
        x = x + 0.1
        x = torch.sign(x)
    elif zero_to_p1 == False:
        x = x - 0.1
        x = torch.sign(x)
    else:
        raise Exception()
    x = convert_n1p1_to_01(x)

    return x


def convert_0lp1_to_01(x):
    '''
    convert x from zeros-positive-number to zero-one
    x can be {0, R>=1}
    x should be float
    '''
    assert x.dtype == torch.float32
    x = x - 0.1
    x = torch.sign(x)
    x = convert_n1p1_to_01(x)

    return x

