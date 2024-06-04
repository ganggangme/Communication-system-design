'''
Module "my_dtmf"
- dtmf_gen(), digits_gen(), mgz(), dtmf_det(), dtmf_dec()
- ver-1.0(2023.06)
'''

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from numpy.random import normal, randint, choice

FS = 8000
NFRM = 200 # 신호 전체 길이보다는 작게 얼마나 자르는지
DTMF_KEYS = '123456789*0#' ## 12개를 문자열ro...
DTMF_FREQ = np.array([[697, 770, 852, 941], [1209, 1336, 1477, 1633]])  ## 문자yeol
INVLD = 'x'
PWTHR = 100

sd.default.samplerate = FS
sd.default.channels = 1
sd.default.blocksize = NFRM

def dtmf_gen(c, T):
    '''
    DTMF signal generation
    c: digit
    T: duration, msec
    '''
    
    t = np.arange(0, T/1000 ,1/FS)
    m, n = divmod(DTMF_KEYS.index(c), 3)
    x = np.sin(2*np.pi*DTMF_FREQ[0, m]*t) + np.sin(2*np.pi*DTMF_FREQ[1, n]*t)
    return x/2

def digits_gen(digits, Ton, Toff):
    '''
    DTMF signal generation with digits
    '''
    
    x = []
    for c in digits:
        if c in DTMF_KEYS:
            x += dtmf_gen(c, Ton).tolist()
            x += [0] * int(FS * Toff / 1000) ## 정수 곱하면 0 확장 and 리스트 연결 
    return np.array(x)

def mgz(x, k): # 주파수 있는 거 하나만 겟 하는 거
    '''
    Modified Geortzel algorithm for frequency component detection
    '''
    
    N = len(x)
    c = 2 * np.cos( 2* np.pi * k / N)
    v = np.zeros(3) # v[n], v[n-1], v[n-2]
    for n in range(N):
        v[2] = v[1]
        v[1] = v[0]
        v[0] = c * v[1] - v[2] + x[n]
    return v[0]**2 + v[1]**2 - c * v[0] * v[1]

def dtmf_det(x, ratio=0.8):
    '''
    DTMF digit detection, return INVLD if non-DTMF digit
    '''
    
    ff = DTMF_FREQ.reshape(-1)
    kf = ff / FS * len(x)
    xp = np.array([mgz(x, k) for k in kf]).reshape(2, -1)
    ir, ic = xp.argmax(-1) # 가장 큰 값(위상)의 인덱스(위상) 반환

    if xp.sum() > PWTHR and ic != 3 and xp[0, ir] > ratio * xp[0].sum() and xp[1, ic] > ratio * xp[1].sum(): ## 잡음과 비교 
        return DTMF_KEYS[3*ir + ic]
    else:
        return INVLD

def dtmf_dec(x):
    '''
    DTMF decoding
    '''
    
    d = ''
    s = [INVLD] * 3 # s[n-2], s[n-1], s[n]
    for i in range(len(x) // NFRM):
        s[0] = s[1]
        s[1] = s[2] # 한 칸씩 움직여서 계산해요(밀어요)
        s[2] = dtmf_det(x[i*NFRM:(i+1)*NFRM])
        if s[0] == INVLD and s[1] != INVLD and s[2] == s[1]:
            d += s[2]
    return d

