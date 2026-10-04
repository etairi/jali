#!/usr/bin/env python3
"""Independent Python oracles. Requires pycryptodome and mpmath; no Rust imports.

The Gaussian vectors use the base tables that half_gaussian_cdf.py computes from their
definition. The challenge vectors are draws of `rand::autostable` with the exact operator-norm
test of LNP22 section 2.7 (`rand::challenge`): the l1 norm of every draw, so that the rejected
ones are checked too.

Usage: python3 tools/kat/generate.py [--count 100000] [--output primitives.json]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from Crypto.Cipher import AES
import mpmath as mp

from half_gaussian_cdf import tables as cdf_tables

mp.mp.prec = 300
TABLES = cdf_tables()
HALF_3_1, HALF_1_55 = TABLES['HALF_3_1'], TABLES['HALF_1_55']

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

def gaussian(stream, t, n):
    """Width 1.55*2^t. t = 0: a sign bit and K from HALF_1_55, (sign 1, K = 0) rejected, output
    K or -K. t >= 1: all offsets first, t-1 bits each; then per coefficient, until accepted: a
    sign bit b, K from HALF_3_1, k = K+1 (b = 1) or -K (b = 0), and a 192-bit uniform value
    compared with floor(2^192 exp(-x)), x = ((k s - u)^2 - ((k - b) s)^2) / (2 sigma^2),
    s = 2^(t-1), sigma^2 = 961 4^t / 400; output k s - u."""
    r = max(t - 1, 0)
    offsets = int.from_bytes(stream.read((r * n + 7) // 8), 'little')
    signs, remaining = 0, 0
    out = []
    for j in range(n):
        u = (offsets >> (r * j)) & ((1 << r) - 1)
        while True:
            if not remaining:
                signs = int.from_bytes(stream.read(8), 'little')
                remaining = 64
            b = signs & 1
            signs >>= 1
            remaining -= 1
            value = int.from_bytes(stream.read(32), 'little')
            if t == 0:
                k = sum(value < x for x in HALF_1_55)
                if b == 1 and k == 0:
                    continue
                out.append(str(k if b == 1 else -k))
                break
            k = sum(value < x for x in HALF_3_1)
            k = k + 1 if b else -k
            s = 2 ** (t - 1)
            numerator = ((k * s - u) ** 2 - ((k - b) * s) ** 2) * 200
            denominator = 961 * 2 ** (2 * t)
            threshold = int(mp.exp(-mp.mpf(numerator) / denominator) * 2 ** 192)
            if int.from_bytes(stream.read(24), 'little') < threshold:
                out.append(str(k * s - u))
                break
    return out

def autostable(stream, d, omega):
    first = [x - omega for x in uniform(stream, 2 * omega + 1, d // 2)]
    c = [0] * d
    c[: d // 2] = first
    for i in range(1, d // 2):
        c[d - i] = -first[i]
    return c

def negacyclic_mul(a, b, d):
    out = [0] * d
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if i + j < d:
                    out[i + j] += x * y
                else:
                    out[i + j - d] -= x * y
    return out

def eta_l1(c, d):
    """|| (sigma_{-1}(c) c)^32 ||_1, exactly."""
    s = [c[0]] + [-c[d - i] for i in range(1, d)]
    p = negacyclic_mul(s, c, d)
    for _ in range(5):
        p = negacyclic_mul(p, p, d)
    return sum(abs(x) for x in p)

def challenge(stream, d, omega, eta, limit=64):
    """The first draw of autostable whose eta_l1 is at most eta^64; the l1 norms of all draws."""
    norms = []
    for _ in range(limit):
        c = autostable(stream, d, omega)
        n = eta_l1(c, d)
        norms.append(n)
        if n <= eta ** 64:
            return c, norms
    raise ValueError("no challenge within eta")

def challenges(seed):
    """For both proof degrees: the first two domains whose first draw is rejected, and domains 0
    and 1."""
    out = []
    for d, omega, eta in ((64, 8, 140), (128, 2, 59)):
        picked, domain, rejected = [], 0, 0
        while rejected < 2 or domain < 2:
            c, norms = challenge(Stream(seed, domain), d, omega, eta)
            if domain < 2 or (len(norms) > 1 and rejected < 2):
                picked.append(dict(degree=d, omega=omega, eta=eta, domain=str(domain),
                                   norms=[str(n) for n in norms], challenge=c))
                rejected += len(norms) > 1
            domain += 1
        out.extend(picked)
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

def coin(rnd, low, prob, i):
    """The 256-bit coin of vector i. It draws from rnd what the 128-bit coins drew, so every
    vector keeps the inputs it had with them. i%3!=0: that 128-bit coin as the high half and a
    uniform low half from `low`. i%3==0: the threshold prob*2^256 offset by -2^112, +2^100 or
    +2^112. These offsets are far above the error of the 192-bit exponentials there (below
    2^90 units of 2^-256) and far below 2^128: a test that resolved its coin to 128 bits would
    accept most of the coins above the threshold."""
    if i%3==0:
        offset={-1:-2**112,0:2**100,1:2**112}[rnd.choice([-1,0,1])]
        return max(0,min(2**256-1,int(mp.floor(prob*2**256))+offset))
    return (rnd.randrange(2**128)<<128)|low.getrandbits(128)

def rational_rejection(count):
    """Decisions for a fractional variance n/d: the sampler's 961*4^t/400 or a random fraction."""
    rnd = random.Random(20260928)
    low = random.Random(20261003)
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
        u=coin(rnd,low,prob,i)
        accept=(u <= prob*2**256) and not (policy==1 and dot<0)
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
    low = random.Random(20261002)
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
        u=coin(rnd,low,prob,i)
        accept=(u <= prob*2**256) and not (policy==1 and dot<0)
        rejection.append(dict(policy=policy,dot=str(dot),norm=str(norm),variance=str(variance),m_scaled=str(m_scaled),u=str(u),accept=bool(accept)))
    exps=[]
    for i in range(128):
        n=rnd.randrange(0,1<<200); d=rnd.randrange(1,1<<200)
        exps.append(dict(n=str(n),d=str(d),scaled=str(int(mp.exp(-mp.mpf(n)/d)*2**192))))
    gaussians=[dict(t=t,values=gaussian(Stream(seed,7),t,64)) for t in [0,1,2,6,24,29,60,100]]
    return dict(seed=seed.hex(),prgs=prgs,samplers=samplers,binomial=binomial(Stream(seed,7),3,137),gaussians=gaussians,
                codec=codec_vector(),challenges=challenges(seed),rejection=rejection,rational_rejection=rational_rejection(count),
                exponentials=exps)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=512)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'kat'/'primitives.json')
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(generate(args.count),indent=2)+'\n')
