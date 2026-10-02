#!/usr/bin/env python3
"""Parameter derivation with a provenance-bearing MLWE report.

Pure Python, mpmath and sympy; no C build and no implicit network access. The positional form
(request, report, --output) derives a set at the request's modulus from the MLWE rank and
delta of the report, which must refer to the same modulus and degree; it makes no hardness
estimate of its own. `rank --mlwe-report` writes such a report from the tool's uSVP estimate.

The subcommands derive, regenerate, check, prime, rank and xcheck run the parameter tool in
`jali_params` (hardness estimates, modulus search, placement search); see tools/README.md.
"""
import argparse
import json
import math
from pathlib import Path
import mpmath as mp
import sympy

mp.mp.prec=300

def derive(request,report):
    d=request['degree']
    factors=request['prime_factors']
    q=math.prod(factors)
    if d not in (64,128) or len(factors) not in (1,2) or any(p%8!=5 or not sympy.isprime(p) for p in factors):
        raise ValueError('invalid proof ring')
    if sorted(set(factors))!=factors:
        raise ValueError('prime factors must be strictly ascending')
    if report['degree']!=d or report['modulus']!=q or not 1 <= report['delta'] <= 1.0044 or not report['provenance']:
        raise ValueError('MLWE report must match this ring and target')
    omega,eta=(8,140) if d==64 else (2,59)
    # TboxParams::check takes log2 of the f64 nearest q1; the exact logarithm differs from it
    # for 2^64-46080 <= q1 < 2^64 (lambda 4 instead of 2), and Rust would refuse the set.
    lam=2
    log_q1=math.log2(float(factors[0]))
    while lam*log_q1<128.0:
        lam+=2
    z=len(request['l2_rows'])
    if len(request['l2_bounds_squared'])!=z:
        raise ValueError('mismatched exact blocks')
    nex=sum(request['l2_rows'])+request['n_bin']+z
    # The sign proof of the range blocks needs Z_q to be a field (see TboxParams::check).
    if len(factors)>1 and (nex or request['n_prime']):
        raise ValueError('binary, exact-norm and range blocks require a prime modulus')
    eslots=256//d if nex else 0
    dslots=256//d if request['n_prime'] else 0
    lext=eslots+dslots+1+lam//2+1
    def rounded(value):
        if value==0:
            return 0
        t=max(0,int(mp.floor(mp.log(value/mp.mpf('1.55'),2))))
        return t if abs(mp.mpf('1.55')*2**t-value)<=abs(mp.mpf('1.55')*2**(t+1)-value) else t+1
    logs=[rounded(14*eta*mp.sqrt(request['alpha_squared']+z*d)),0,
          rounded(5*mp.sqrt(337)*mp.sqrt(sum(request['l2_bounds_squared'])+(request['n_bin']+z)*d)),
          # Euclidean bound sqrt(n'd)*linf on the approximate-range vector (LNP22 Fig. 10).
          rounded(5*mp.sqrt(337)*mp.sqrt(request['n_prime']*d)*request['linf_bound'])]
    def widths():
        return [mp.mpf('1.55')*2**t for t in logs]
    def bound(n,m2,D,gamma):
        s1,s2,_,_=widths()
        b=s2*mp.sqrt(2*m2*d)+eta*mp.mpf(2)**(D-1)*mp.sqrt(n*d)+gamma*mp.sqrt(n*d)/2
        b1=2*s1*mp.sqrt(2*(request['m1']+z)*d)
        return 4*eta*mp.sqrt(b1*b1+4*b*b)
    def hard(n,m2,D,gamma):
        b=bound(n,m2,D,gamma)
        delta=mp.mpf(2)**(mp.log(b,2)**2/(4*n*d*mp.log(q,2)))
        return delta<mp.mpf('1.0044') and b<q
    n=1
    while True:
        m2=report['rank']+n+request['l']+lext
        logs[1]=rounded(eta*mp.sqrt(m2*d))
        if hard(n,m2,0,0):
            break
        n+=1
        if n>65535:
            raise ValueError('MSIS search exhausted')
    gamma=2**(q.bit_length()-1)
    while not hard(n,m2,0,gamma):
        gamma//=2
    candidates=[int(x) for x in sympy.divisors(q-1) if x%2==0 and 4*gamma < 5*x <= 5*gamma]
    if not candidates:
        raise ValueError('no suitable divisor; choose another prime and rerun the estimator')
    gamma=max(candidates)
    D=q.bit_length()-1
    while not (hard(n,m2,D,gamma) and mp.mpf(2)**(D-1)*omega*d<gamma):
        D-=1
        if D<0:
            raise ValueError('no compression exponent')
    out=dict(request)
    out.update(m2=m2,n_msis=n,log_sigma=logs,gamma=gamma,d_bits=D,mlwe_rank=report['rank'],mlwe_delta=report['delta'],estimator=report['provenance'])
    return out

if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1] in ('derive','regenerate','check','prime','rank',
                                           'xcheck'):
        from jali_params.cli import main
        main(sys.argv[1:])
        sys.exit(0)
    parser=argparse.ArgumentParser()
    parser.add_argument('request',type=Path)
    parser.add_argument('mlwe_report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    params=derive(json.loads(args.request.read_text()),json.loads(args.mlwe_report.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(params,indent=2)+'\n')
