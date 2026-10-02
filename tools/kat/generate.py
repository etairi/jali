#!/usr/bin/env python3
"""Independent Python oracles. Requires pycryptodome and mpmath; no Rust imports.

The Gaussian vectors use the base table that half_gaussian_cdf.py computes from its definition.

Usage: python3 tools/kat/generate.py [--count 100000] [--output primitives.json]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from Crypto.Cipher import AES
import mpmath as mp

from half_gaussian_cdf import table as cdf_table

mp.mp.prec = 300
CDF = cdf_table()

class Stream:
    def __init__(self, seed, domain):
        iv = domain.to_bytes(8, 'little') + bytes(8)
        self.aes = AES.new(seed, AES.MODE_ECB)
        self.counter = int.from_bytes(iv, 'big')
        self.buffer = b''
    def read(self, n):
        while len(self.buffer) < n:
            self.buffer += self.aes.encrypt(self.counter.to_bytes(16, 'big'))
            self.counter += 1
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

def uniform(stream, modulus, count):
    width = (modulus - 1).bit_length()
    out = []
    while len(out) < count:
        n = count - len(out)
        chunk = int.from_bytes(stream.read((width*n + 7)//8), 'little')
        out += [x for i in range(n) if (x := (chunk >> (i*width)) & ((1 << width)-1)) < modulus]
    return out

def binomial(stream, k, n):
    bits = int.from_bytes(stream.read((2*k*n+7)//8), 'little')
    mask = (1 << k)-1
    return [((bits >> (i*k)) & mask).bit_count() - ((bits >> (k*n+i*k)) & mask).bit_count() for i in range(n)]

def gaussian(stream,t,n):
    cdf=CDF
    offsets=int.from_bytes(stream.read((t*n+7)//8),'little')
    signs,remaining=0,0
    out=[]
    for j in range(n):
        u=(offsets>>(t*j))&((1<<t)-1)
        while True:
            if not remaining:
                signs=int.from_bytes(stream.read(8),'little')
                remaining=64
            b=signs&1
            signs>>=1
            remaining-=1
            value=(int.from_bytes(stream.read(8),'little')<<64)|int.from_bytes(stream.read(8),'little')
            k=sum(value<x for x in cdf)
            k=k+1 if b else -k
            numerator=((k*2**t-u)**2-((k-b)*2**t)**2)*200
            denominator=961*2**(2*t)
            threshold=int(mp.exp(-mp.mpf(numerator)/denominator)*2**192)
            if int.from_bytes(stream.read(24),'little')<threshold:
                out.append(str(k*2**t-u))
                break
    return out

def codec_vector():
    bits = []
    def put(n, width):
        bits.extend((n >> i)&1 for i in range(width))
    u = [0, 1, 12, 4, 10]
    for v in u:
        put(v, 4)
    g = [-1000, -17, -16, -15, -1, 0, 1, 15, 16, 17, 1000]
    for v in g:
        low = (v+16)%32-16
        high = (v-low)//32
        n = -2*high if high <= 0 else 2*high-1
        bits.extend([1]*n + [0])
        put(low%32,5)
    hints = [0, 1, -1, 2, -2, 3, -3, 16, -16]
    for h in hints:
        if h in [0,1,-1]:
            bits.extend({0:[0,0],1:[0,1],-1:[1,0]}[h])
        else:
            bits.extend([1,1]+[0]*(2*abs(h)-(4 if h>0 else 3))+[1])
    bits.append(1)
    bits.extend([0]*((-len(bits))%8))
    encoded=bytes(sum(bits[j+i]<<i for i in range(8)) for j in range(0,len(bits),8))
    return dict(uniform=u,gaussian=g,hints=hints,encoded=encoded.hex())

def rational_rejection(count):
    """Decisions for a fractional variance n/d: the sampler's 961*4^t/400 or a random fraction."""
    rnd = random.Random(20260928)
    out=[]
    for i in range(count):
        if (i//4)%2==0:
            num,den=961 << (2*rnd.randrange(0,101)),400
        else:
            num=rnd.randrange(1,1 << (8+i%190)); den=rnd.randrange(1,1 << rnd.choice([1,16,64]))
        size=max(1,num//den)
        norm=rnd.randrange(0,2*size)
        dot=rnd.randrange(-4*size,4*size)
        m_scaled=rnd.randrange(1<<128,8<<128)
        # Standard, Rej_2, bimodal; each meets the boundary values of u (i%3==0).
        policy=(i//3)%3
        # x/s^2 = x*den/num, evaluated without rounding the variance.
        if policy<2:
            prob=mp.exp(mp.mpf((norm-2*dot)*den)/(2*num))*2**128/m_scaled
            if policy==1 and dot<0:
                prob=mp.mpf(0)
        else:
            prob=2**128/(m_scaled*mp.exp(-mp.mpf(norm*den)/(2*num))*mp.cosh(mp.mpf(dot*den)/num))
        prob=min(mp.mpf(1),prob)
        if i%3==0:
            u=max(0,min(2**128-1,int(mp.floor(prob*2**128))+rnd.choice([-1,0,1])))
        else:
            u=rnd.randrange(2**128)
        accept=(u <= prob*2**128) and not (policy==1 and dot<0)
        out.append(dict(policy=policy,dot=str(dot),norm=str(norm),variance_numerator=str(num),
                        variance_denominator=str(den),m_scaled=str(m_scaled),u=str(u),accept=bool(accept)))
    return out

def generate(count):
    seed = bytes(range(32))
    domains = [0,1,2,256,(0x12345678 << 32)|0xabcdef01]
    prgs = [dict(domain=str(d), aes=Stream(seed,d).read(103).hex(),
                 shake=hashlib.shake_128(seed+d.to_bytes(8,'little')).digest(103).hex()) for d in domains]
    # The last five moduli, appended for 256-bit coefficients, straddle 2^128 and reach 2^256.
    samplers = [dict(modulus=str(q),values=[str(x) for x in uniform(Stream(seed,7),q,137)])
                for q in [13,1099511627917,(1<<61)+15,(1<<80)+7,
                          (1<<128)-1,1<<128,(1<<128)+165,(1<<240)+325,(1<<256)-435]]
    rejection=[]
    rnd = random.Random(20260926)
    for i in range(count):
        variance=rnd.randrange(1,1 << (8+i%190))
        norm=rnd.randrange(0,2*variance)
        dot=rnd.randrange(-4*variance,4*variance)
        m_scaled=rnd.randrange(1<<128,8<<128)
        # Standard, Rej_2, bimodal; each meets the boundary values of u (i%3==0).
        policy=(i//3)%3
        if policy<2:
            prob=mp.exp(mp.mpf(norm-2*dot)/(2*variance))*2**128/m_scaled
            if policy==1 and dot<0:
                prob=mp.mpf(0)
        else:
            prob=2**128/(m_scaled*mp.exp(-mp.mpf(norm)/(2*variance))*mp.cosh(mp.mpf(dot)/variance))
        prob=min(mp.mpf(1),prob)
        if i%3==0:
            u=max(0,min(2**128-1,int(mp.floor(prob*2**128))+rnd.choice([-1,0,1])))
        else:
            u=rnd.randrange(2**128)
        accept=(u <= prob*2**128) and not (policy==1 and dot<0)
        rejection.append(dict(policy=policy,dot=str(dot),norm=str(norm),variance=str(variance),m_scaled=str(m_scaled),u=str(u),accept=bool(accept)))
    exps=[]
    for i in range(128):
        n=rnd.randrange(0,1<<200); d=rnd.randrange(1,1<<200)
        exps.append(dict(n=str(n),d=str(d),scaled=str(int(mp.exp(-mp.mpf(n)/d)*2**192))))
    gaussians=[dict(t=t,values=gaussian(Stream(seed,7),t,64)) for t in [0,1,6,24,29,60,100]]
    return dict(seed=seed.hex(),prgs=prgs,samplers=samplers,binomial=binomial(Stream(seed,7),3,137),gaussians=gaussians,
                codec=codec_vector(),rejection=rejection,rational_rejection=rational_rejection(count),
                exponentials=exps)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=512)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'kat'/'primitives.json')
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(generate(args.count),indent=2)+'\n')
