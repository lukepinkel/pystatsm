# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:33:40 2021

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.stats

TWOPI = 2.0 * np.pi
LN2PI = np.log(TWOPI)

class SASH:
    
    @staticmethod
    def loglike(y, mu, sigma, nu, tau):
        return _logpdf(y, mu, sigma, nu, tau)
    
    @staticmethod
    def d1loglike(y, mu, sigma, nu, tau):
        return _d1logpdf(y, mu, sigma, nu, tau)
    
    @staticmethod
    def d2loglike(y, mu, sigma, nu, tau):
        return _d2logpdf(y, mu, sigma, nu, tau)

    @staticmethod
    def d3loglike(y, mu, sigma, nu, tau):
        return _d3logpdf(y, mu, sigma, nu, tau)

    @staticmethod
    def d4loglike(y, mu, sigma, nu, tau):
        return _d4logpdf(y, mu, sigma, nu, tau)

    
    

def _expand_arrs(*arrs):
    y = np.concatenate([np.expand_dims(np.asarray(arr), -1) for arr in arrs], axis=-1)
    return y


def _logpdf(x, m=0.0, s=1.0, v=0.0, t=1.0):
    z = (x - m) / s
    y = np.arcsinh(z) * t - v
    r, c, w = np.sinh(y), np.cosh(y), np.sqrt(1.0 + z**2)
    logp = np.log(t) + np.log(c) - np.log(s) - np.log(w) - r**2 / 2.0 - LN2PI / 2.0
    return logp


def _d1logpdf(x, m, s, v, t):
    si1 = 1.0 / s
    si2 = si1 / s
    r = x - m
    r2 = r**2
    rs = si2*r2
    z = (rs + 1.0)**(-1)
    u = np.arcsinh(r*si1)
    y = t*u - v
    x8 = np.sinh(y)
    x9 = np.cosh(y)
    x10 = x8*x9
    x11 = t/np.sqrt(rs + 1)
    x12 = x11*si1
    x13 = x8/x9
    x14 = si2*r*x11
    m = -1/2*si2*z*(2*m - 2*x) + x10*x12 - x12*x13
    s = x10*x14 - x13*x14 - si1 + r2*z/s**3
    v = x10 - x13
    t = -x10*u + x13*u + t**(-1)
    
    D1 = _expand_arrs(m, s, v, t)
    return D1

def _d2logpdf(x, m, s, v, t):
    x0 = s**(-2)
    x1 = m - x
    x2 = x0*x1**2
    x3 = x2 + 1.0
    x4 = x3**(-1)
    x5 = t**2
    x6 = x2 + 1
    x7 = x5/x6
    x8 = 2/x3**2
    x9 = s**(-1)
    x10 = x1*x9
    x11 = np.arcsinh(x10)
    x12 = t*x11
    x13 = v + x12
    x14 = np.sinh(x13)
    x15 = x14**2
    x16 = x15*x7
    x17 = np.cosh(x13)
    x18 = x17**2
    x19 = x18*x7
    x20 = x15/x18
    x21 = x20*x7
    x22 = x14*x17
    x23 = t/x6**(3/2)
    x24 = x10*x23
    x25 = x14/x17
    x26 = 2*x10
    x27 = x1**3/s**3
    x28 = 1/np.sqrt(x6)
    x29 = t*x28
    x30 = x22*x29
    x31 = x25*x29
    x32 = x2*x23
    x33 = -x15 - x18 - x20 + 1
    x34 = x28*x9
    x35 = x12*x15
    x36 = x12*x18
    x37 = x12*x20
    x38 = x23*x27
    x39 = x0*x1
    x40 = x11**2
    mm = x0*(-x16 - x19 + x2*x8 - x21 + x22*x24 - x24*x25 - x4 + x7)
    ms = x0*(x10*x16 + x10*x19 + x10*x21 - x10*x7 - x22*x32 + x25*x32 + x26*x4 - x27*x8 + x30 - x31)
    mv = t*x33*x34
    mt = x34*(x12 - x22 + x25 - x35 - x36 - x37)
    ss = x0*(-x16*x2 - x19*x2 - x2*x21 - 3*x2*x4 + x2*x7 + x22*x38 - x25*x38 - x26*x30 + x26*x31 + 1 + x1**4*x8/s**4)
    sv = x29*x39*(x15 + x18 + x20 - 1)
    st = x28*x39*(-x12 + x22 - x25 + x35 + x36 + x37)
    vv = x33
    vt = x11*x33
    tt = -x15*x40 - x18*x40 - x20*x40 + x40 - 1/x5
    D2 = _expand_arrs(mm, ms, mv, mt, ss, sv, st, vv, vt, tt)
    return D2
    
def _d3logpdf(x, m, s, v, t):
    x0 = s**(-3)
    x1 = s**(-2)
    x2 = m - x
    x3 = x1*x2**2
    x4 = x3 + 1.0
    x5 = x4**(-2)
    x6 = s**(-1)
    x7 = x2*x6
    x8 = 6*x7
    x9 = 8/x4**3
    x10 = x0*x2**3
    x11 = t**2
    x12 = x11*x7
    x13 = x3 + 1
    x14 = 3/x13**2
    x15 = x12*x14
    x16 = np.arcsinh(x7)
    x17 = t*x16
    x18 = v + x17
    x19 = np.sinh(x18)
    x20 = x19**2
    x21 = np.cosh(x18)
    x22 = x21**2
    x23 = x13**(-3/2)
    x24 = t*x23
    x25 = x19*x21
    x26 = x24*x25
    x27 = x21**(-1)
    x28 = x19*x27
    x29 = x24*x28
    x30 = t**3
    x31 = x23*x30
    x32 = 4*x25
    x33 = x31*x32
    x34 = 2*x19
    x35 = x27*x34
    x36 = x31*x35
    x37 = x21**(-3)
    x38 = x19**3*x37
    x39 = 2*x38
    x40 = x31*x39
    x41 = 3*t
    x42 = x41/x13**(5/2)
    x43 = x3*x42
    x44 = x22**(-1)
    x45 = x20*x44
    x46 = x4**(-1)
    x47 = x13**(-1)
    x48 = 2*x47
    x49 = x11*x48
    x50 = x2**4/s**4
    x51 = x11*x14
    x52 = x3*x51
    x53 = x23*x7
    x54 = x25*x53
    x55 = x10*x42
    x56 = x28*x53
    x57 = x30*x53
    x58 = x20*x53
    x59 = x22*x53
    x60 = x32*x47
    x61 = t*x60
    x62 = t*x48
    x63 = x28*x62
    x64 = x38*x62
    x65 = x45*x53
    x66 = t*x1
    x67 = x20*x62
    x68 = x22*x62
    x69 = x45*x62
    x70 = x16*x60
    x71 = x16*x49
    x72 = x28*x71
    x73 = x38*x71
    x74 = x2**5/s**5
    x75 = 4*x12*x47
    x76 = x10*x51
    x77 = 2*x21
    x78 = 1/np.sqrt(x13)
    x79 = x19*x78
    x80 = x77*x79
    x81 = 2*x78
    x82 = x28*x81
    x83 = x42*x50
    x84 = x23*x3
    x85 = x25*x84
    x86 = 5*t
    x87 = x28*x84
    x88 = x30*x84
    x89 = x20*x78
    x90 = x22*x78
    x91 = x20*x84
    x92 = x22*x84
    x93 = x44*x89
    x94 = x61*x7
    x95 = x63*x7
    x96 = x64*x7
    x97 = x45*x84
    x98 = x62*x7
    x99 = x17*x84
    x100 = x67*x7
    x101 = x68*x7
    x102 = x21*x79
    x103 = x27*x79
    x104 = x17*x91
    x105 = x17*x92
    x106 = x69*x7
    x107 = x12*x70
    x108 = x7*x72
    x109 = x17*x97
    x110 = x7*x73
    x111 = x20*x37
    x112 = x111 - x27 - x77
    x113 = x6*x81
    x114 = x17*x32
    x115 = x17*x35
    x116 = x17*x38
    x117 = 2*x116
    x118 = -x20 - x22 - x45 + 1
    x119 = x17*x28
    x120 = x19*x77
    x121 = x120*x17
    x122 = x50*x51
    x123 = 6*x11*x3*x47
    x124 = t*x8
    x125 = x42*x74
    x126 = 7*x10
    x127 = 2*x89
    x128 = 2*x90
    x129 = 2*x93
    x130 = x0*x2
    x131 = x20 + x22 + x45 - 1
    x132 = x1*x2
    x133 = x112*x34
    x134 = x16**3
    mmm = x0*(-x10*x9 + x15*x20 + x15*x22 + x15*x45 - x15 - x25*x43 + x26 + x28*x43 - x29 - x33 - x36 + x40 + x5*x8)
    mms = x0*(x20*x49 - x20*x52 + x22*x49 - x22*x52 + x25*x55 - x28*x55 - 10*x3*x5 + x32*x57 + x35*x57 - x39*x57 - x41*x54 + x41*x56 + x45*x49 - x45*x52 + 2*x46 - x49 + x50*x9 + x52)
    mmv = x66*(-x53 + x58 + x59 - x61 - x63 + x64 + x65)
    mmt = x1*(-x11*x70 - x17*x53 + x17*x58 + x17*x59 + x17*x65 + x54 - x56 + x62 - x67 - x68 - x69 - x72 + x73)
    mss = x0*(-t*x80 + t*x82 + 14*x10*x5 - x20*x75 + x20*x76 - x22*x75 + x22*x76 - x25*x83 + x28*x83 - x32*x88 - x35*x88 + x39*x88 - x45*x75 + x45*x76 - x46*x8 - x74*x9 + x75 - x76 + x85*x86 - x86*x87)
    msv = x66*(-x78 + x84 + x89 + x90 - x91 - x92 + x93 + x94 + x95 - x96 - x97)
    mst = x1*(x100 + x101 + x102 - x103 - x104 - x105 + x106 + x107 + x108 - x109 - x110 - x17*x78 + x17*x89 + x17*x90 + x17*x93 - x85 + x87 - x98 + x99)
    mvv = t*x112*x113*x19
    mvt = x6*x78*(-x114 - x115 + x117 + x118)
    mtt = x113*x16*(x116 + x118 - x119 - x121)
    sss = x0*(x10*x33 + x10*x36 - x10*x40 + x102*x124 - x103*x124 - x122*x20 - x122*x22 - x122*x45 + x122 + x123*x20 + x123*x22 + x123*x45 - x123 + x125*x25 - x125*x28 - x126*x26 + x126*x29 + 12*x3*x46 - 18*x5*x50 - 2 + x2**6*x9/s**6)
    ssv = t*x130*(-x127 - x128 - x129 + x81 - x84 + x91 + x92 - x94 - x95 + x96 + x97)
    sst = x130*(-x100 - x101 + x104 + x105 - x106 - x107 - x108 + x109 + x110 - x127*x17 - x128*x17 - x129*x17 + x17*x81 - x80 + x82 + x85 - x87 + x98 - x99)
    svv = x19*x2*x66*x81*(-x111 + x27 + x77)
    svt = x132*x78*(x114 + x115 - x117 + x131)
    stt = x132*x16*x81*(-x116 + x119 + x121 + x131)
    vvv = x133
    vvt = x133*x16
    vtt = x133*x16**2
    ttt = -2*x120*x134 - 2*x134*x28 + 2*x134*x38 + 2/x30
    D3 = _expand_arrs(mmm,mms, mmv, mmt, mss, msv, mst, mvv, mvt, mtt, sss, 
                      ssv, sst, svv, svt, stt, vvv, vvt, vtt, ttt)
    return D3


def _d4logpdf(x, m, s, v, t):
    x0 = s**(-4)
    x1 = s**(-2)
    x2 = m - x
    x3 = x1*x2**2
    x4 = x3 + 1.0
    x5 = x4**(-2)
    x6 = t**2
    x7 = x3 + 1
    x8 = x7**(-2)
    x9 = 4*x8
    x10 = x6*x9
    x11 = t**4
    x12 = x11*x8
    x13 = 2*x12
    x14 = 48/x4**4
    x15 = x0*x2**4
    x16 = x4**(-3)
    x17 = x3*x6
    x18 = 15/x7**3
    x19 = x17*x18
    x20 = s**(-1)
    x21 = x2*x20
    x22 = np.arcsinh(x21)
    x23 = t*x22
    x24 = v + x23
    x25 = np.sinh(x24)
    x26 = x25**2
    x27 = np.cosh(x24)
    x28 = x27**2
    x29 = x11*x9
    x30 = x26*x29
    x31 = x28*x29
    x32 = x28**(-1)
    x33 = x26*x32
    x34 = 8*x33
    x35 = x12*x34
    x36 = x25**4/x27**4
    x37 = 6*x36
    x38 = x12*x37
    x39 = x25*x27
    x40 = t*x39
    x41 = x7**(-5/2)
    x42 = x21*x41
    x43 = 9*x42
    x44 = 15/x7**(7/2)
    x45 = x2**3
    x46 = s**(-3)
    x47 = t*x46
    x48 = x45*x47
    x49 = x39*x48
    x50 = x25/x27
    x51 = t*x50
    x52 = t**3
    x53 = 24*x21
    x54 = x52*x53
    x55 = x39*x41
    x56 = x48*x50
    x57 = 12*x52
    x58 = x42*x57
    x59 = x25**3/x27**3
    x60 = x2**5/s**5
    x61 = x45*x46
    x62 = x21*x8
    x63 = x6*x62
    x64 = 13*x63
    x65 = x18*x6
    x66 = x61*x65
    x67 = x7**(-3/2)
    x68 = x39*x67
    x69 = 3*t
    x70 = x50*x67
    x71 = 12*x68
    x72 = 6*x52
    x73 = x59*x67
    x74 = x15*x44
    x75 = x3*x41
    x76 = 18*x75
    x77 = 24*x52
    x78 = x55*x77
    x79 = x57*x75
    x80 = 2*x67
    x81 = x6*x80
    x82 = 3*x41
    x83 = x3*x82
    x84 = x26*x67
    x85 = x28*x67
    x86 = 4*x6
    x87 = x84*x86
    x88 = x85*x86
    x89 = x26*x83
    x90 = x28*x83
    x91 = x32*x84
    x92 = 8*x91
    x93 = x6*x92
    x94 = 6*x6
    x95 = x36*x94
    x96 = 12*x40
    x97 = 6*t
    x98 = x62*x97
    x99 = x33*x83
    x100 = x23*x67
    x101 = x22*x52
    x102 = x101*x80
    x103 = x23*x84
    x104 = x23*x85
    x105 = 4*x101
    x106 = x6*x71
    x107 = x70*x94
    x108 = 3*x50
    x109 = x108*x41
    x110 = x23*x91
    x111 = x101*x92
    x112 = x101*x37
    x113 = 12*x39
    x114 = x113*x22
    x115 = x22*x94
    x116 = x115*x62
    x117 = x4**(-1)
    x118 = x7**(-1)
    x119 = x118*x6
    x120 = 6*x119
    x121 = x2**6/s**6
    x122 = x15*x65
    x123 = x17*x8
    x124 = 22*x123
    x125 = x118*x26
    x126 = x118*x28
    x127 = x125*x32
    x128 = x44*x60
    x129 = 27*x41
    x130 = x21*x70
    x131 = x41*x57
    x132 = x131*x50
    x133 = x21*x67
    x134 = x133*x59
    x135 = x59*x61
    x136 = x61*x82
    x137 = x21*x84
    x138 = x21*x85
    x139 = x136*x26
    x140 = x136*x28
    x141 = t*x118
    x142 = 8*x39
    x143 = x141*x142
    x144 = 4*x141
    x145 = x144*x50
    x146 = x144*x59
    x147 = x3*x8
    x148 = x21*x91
    x149 = x147*x97
    x150 = x136*x33
    x151 = 3*x21
    x152 = x144*x33
    x153 = x119*x22
    x154 = 4*x153
    x155 = x115*x147
    x156 = 2*x26
    x157 = x141*x156
    x158 = 2*x28
    x159 = x141*x158
    x160 = 3*x36
    x161 = x39*x80
    x162 = x161*x21
    x163 = 2*x1
    x164 = t*x163
    x165 = 2*x153
    x166 = x22*x86
    x167 = x127*x6
    x168 = 8*x22
    x169 = x120*x22*x36
    x170 = 4*x23
    x171 = x23*x80
    x172 = x21*x50
    x173 = x21*x59
    x174 = x22**2
    x175 = x119*x174
    x176 = 2*x174*x6
    x177 = x118*x23
    x178 = 4*x177
    x179 = x174*x86
    x180 = t*x174
    x181 = x2**7/s**7
    x182 = 18*x21
    x183 = x60*x65
    x184 = x6*x8
    x185 = x184*x61
    x186 = 31*x185
    x187 = x125*x21
    x188 = 18*x6
    x189 = x126*x21
    x190 = 1/np.sqrt(x7)
    x191 = 6*x190
    x192 = x191*x39
    x193 = x191*x50
    x194 = x121*x44
    x195 = 36*x15
    x196 = x40*x41
    x197 = 27*t
    x198 = x3*x68
    x199 = x41*x51
    x200 = x3*x70
    x201 = 36*x3
    x202 = x52*x68
    x203 = 18*x52
    x204 = x131*x59
    x205 = x3*x67
    x206 = x205*x59
    x207 = x15*x82
    x208 = x207*x28
    x209 = x207*x26
    x210 = x3*x81
    x211 = x3*x85
    x212 = x3*x84
    x213 = x3*x88
    x214 = x3*x87
    x215 = x207*x33
    x216 = x3*x91
    x217 = x141*x21
    x218 = 16*x39
    x219 = 8*x217
    x220 = x48*x8
    x221 = 6*x220
    x222 = x221*x59
    x223 = x205*x95
    x224 = x221*x50
    x225 = x17*x92
    x226 = x113*x220
    x227 = 2*x190
    x228 = x227*x28
    x229 = x227*x26
    x230 = x229*x32
    x231 = x227 - x228 - x229 - x230
    x232 = x227*x23
    x233 = x207*x23
    x234 = 5*x3
    x235 = x102*x3
    x236 = x221*x26
    x237 = x221*x28
    x238 = x227*x39
    x239 = x227*x50
    x240 = x209*x23
    x241 = x208*x23
    x242 = x207*x39
    x243 = x105*x212
    x244 = x105*x211
    x245 = x109*x15
    x246 = x17*x71
    x247 = x221*x33
    x248 = x107*x3
    x249 = x206*x94
    x250 = x114*x185
    x251 = 8*x153
    x252 = x215*x23
    x253 = x115*x8
    x254 = x253*x50*x61
    x255 = x135*x253
    x256 = x111*x3
    x257 = x112*x205
    x258 = x157*x21
    x259 = x159*x21
    x260 = x190*x50
    x261 = x190*x59
    x262 = x161*x3
    x263 = x152*x21
    x264 = x160*x217
    x265 = x190*x26
    x266 = x190*x28
    x267 = x165*x21
    x268 = x265*x32
    x269 = x166*x187
    x270 = x166*x189
    x271 = x142*x217
    x272 = 4*x39
    x273 = x190*x272
    x274 = x145*x21
    x275 = x146*x21
    x276 = x170*x198
    x277 = x171*x3
    x278 = x277*x50
    x279 = x277*x59
    x280 = x127*x21
    x281 = x168*x280*x6
    x282 = x169*x21
    x283 = x118*x21
    x284 = x205*x22
    x285 = x175*x21
    x286 = x212*x22
    x287 = x211*x22
    x288 = x176*x187
    x289 = x176*x189
    x290 = x142*x23
    x291 = x283*x290
    x292 = x170*x283
    x293 = x292*x50
    x294 = x180*x262
    x295 = x292*x59
    x296 = x216*x22
    x297 = x180*x200
    x298 = x180*x206
    x299 = x179*x280
    x300 = x160*x285
    x301 = 4*x33
    x302 = -x156 - x158 - x160 + x301 - 1
    x303 = x20*x227
    x304 = 2*x39
    x305 = x158*x23
    x306 = x156*x23
    x307 = x160*x23
    x308 = x23*x301
    x309 = -x23 - x305 - x306 - x307 + x308
    x310 = 2*x50
    x311 = 2*x59
    x312 = x22*x227
    x313 = 6*x39
    x314 = 3*x59
    x315 = x121*x65
    x316 = 40*x15*x184
    x317 = 36*x17
    x318 = x181*x44
    x319 = 45*x60
    x320 = 48*x48
    x321 = x61*x77
    x322 = 6*x265
    x323 = 6*x266
    x324 = 6*x268
    x325 = x39*x53
    x326 = 12*x217
    x327 = x0*x2
    x328 = 7*x3
    x329 = 12*x153
    x330 = x227*x59
    x331 = x2*x46
    x332 = x1*x2
    x333 = x227*x332
    x334 = x23 + x305 + x306 + x307 - x308
    x335 = 2*x302
    x336 = x22**4
    mmmm = x0*(x10*x26 + x10*x28 + x10*x33 - x10 - x13 + x14*x15 - 48*x16*x3 - x19*x26 - x19*x28 - x19*x33 + x19 - x30 - x31 + x35 - x38 - x40*x43 + x43*x51 + x44*x49 - x44*x56 + 6*x5 + x50*x58 + x54*x55 - x58*x59)
    mmms = x0*(x13*x21 - x14*x60 + 72*x16*x61 + x21*x30 + x21*x31 - x21*x35 + x21*x38 - x26*x64 + x26*x66 - x28*x64 + x28*x66 - x3*x78 - x33*x64 + x33*x66 - x40*x74 + x40*x76 - x5*x53 - x50*x79 + x51*x74 - x51*x76 + x52*x71 + x59*x79 + x64 - x66 - x68*x69 + x69*x70 + x70*x72 - x72*x73)
    mmmv = x47*(x50*x98 - x59*x98 + x62*x96 - x67*x95 - x67 - x81 + x83 + x84 + x85 - x87 - x88 - x89 - x90 + x91 + x93 - x99)
    mmmt = x46*(-x100 - x102 + x103 + x104 - x105*x84 - x105*x85 - x106 - x107 + x109*x3 + x110 + x111 - x112*x67 + x114*x63 + x116*x50 - x116*x59 + x23*x83 - x23*x89 - x23*x90 - x23*x99 + x26*x98 + x28*x98 + x33*x98 - x39*x83 + x68 - x70 + x73*x94 - x98)
    mmss = x0*(-12*t*x130 + t*x21*x71 - 6*x117 + x120 + x121*x14 - x122*x26 - x122*x28 - x122*x33 + x122 + x124*x26 + x124*x28 + x124*x33 - x124 - x125*x94 - x126*x94 - x127*x94 + x128*x40 - x128*x51 - x129*x49 + x129*x56 - x13*x3 - x130*x57 - x131*x135 + x132*x61 + x134*x57 - 96*x15*x16 - x3*x30 - x3*x31 + x3*x35 - x3*x38 + 54*x3*x5 - x54*x68 + x61*x78)
    mmsv = x47*(x133*x95 + 3*x133 - x136 - 3*x137 - 3*x138 + x139 + x140 + x143 + x145 - x146 - x147*x96 - 3*x148 - x149*x50 + x149*x59 + x150 + x21*x81 + x21*x87 + x21*x88 - x21*x93)
    mmst = x46*(x100*x151 + x102*x21 - x103*x151 - x104*x151 + x105*x137 + x105*x138 + x106*x21 + x107*x21 - x109*x61 - x110*x151 - x111*x21 + x112*x133 - x114*x123 + 3*x130 - x134*x94 - x136*x23 + x136*x39 + x139*x23 + x140*x23 + x142*x153 + x144*x26 + x144*x28 - x144 - x149*x26 - x149*x28 - x149*x33 + x149 + x150*x23 - x151*x68 + x152 + x154*x50 - x154*x59 - x155*x50 + x155*x59)
    mmvv = x164*(x130 - x134 - x141*x160 - x141 + x152 - x157 - x159 + x162)
    mmvt = x1*(-x125*x166 - x126*x166 - x133 + x137 + x138 - x143 - x145 + x146 + x148 - x165 + x167*x168 - x169 + x170*x21*x68 + x171*x172 - x171*x173)
    mmtt = x163*(x118 - x125*x176 - x125 - x126*x176 - x126 + x127*x179 - x127 + x130*x180 - x133*x22 - x134*x180 + x137*x22 + x138*x22 - x142*x177 + x148*x22 - x160*x175 + x162*x180 - x175 - x178*x50 + x178*x59)
    msss = x0*(t*x192 - t*x193 + x117*x53 - x119*x182 + x13*x61 - x132*x15 - x14*x181 + x15*x204 - x15*x78 + 120*x16*x60 + x167*x182 + x183*x26 + x183*x28 + x183*x33 - x183 - x186*x26 - x186*x28 - x186*x33 + x186 + x187*x188 + x188*x189 - x194*x40 + x194*x51 + x195*x196 - x195*x199 - x197*x198 + x197*x200 + x200*x203 + x201*x202 - x203*x206 + x30*x61 + x31*x61 - x35*x61 + x38*x61 - 96*x5*x61)
    mssv = x47*(-5*x205 + x207 - x208 - x209 - x210 + 5*x211 + 5*x212 - x213 - x214 - x215 + 5*x216 - x217*x218 - x219*x50 + x219*x59 - x222 - x223 + x224 + x225 + x226 + x231)
    msst = x46*(-x100*x234 + x103*x234 + x104*x234 + x110*x234 - x153*x21*x218 - x172*x251 + x173*x251 - 5*x200 - x217*x34 - x219*x26 - x219*x28 + x219 - x221 - x228*x23 - x229*x23 - x23*x230 + x232 + x233 + x234*x68 - x235 + x236 + x237 - x238 + x239 - x240 - x241 - x242 - x243 - x244 + x245 - x246 + x247 - x248 + x249 + x250 - x252 + x254 - x255 + x256 - x257)
    msvv = x164*(-x200 + x206 + x217 + x238 + x258 + x259 + x260 - x261 - x262 - x263 + x264)
    msvt = x1*(-x190 + x205 - x211 - x212 - x216 + x23*x273 + x232*x50 - x232*x59 + x265 + x266 + x267 + x268 + x269 + x270 + x271 + x274 - x275 - x276 - x278 + x279 - x281 + x282)
    mstt = x163*(x180*x238 + x180*x260 - x180*x261 + x187 + x189 - x190*x22 + x22*x265 + x22*x266 + x22*x268 + x280 - x283 + x284 + x285 - x286 - x287 + x288 + x289 + x291 + x293 - x294 - x295 - x296 - x297 + x298 - x299 + x300)
    mvvv = t*x302*x303
    mvvt = x303*(-x304 + x309 - x50 + x59)
    mvtt = x20*x312*(-x272 + x309 - x310 + x311)
    mttt = x174*x303*(-x108 + x309 - x313 + x314)
    ssss = x0*(t*x260*x53 - 60*x117*x3 + x119*x201 - 144*x121*x16 - x125*x317 - x126*x317 - x127*x317 - x13*x15 + x132*x60 - x15*x30 - x15*x31 + x15*x35 - x15*x38 + 150*x15*x5 - x190*x40*x53 - x196*x319 + x199*x319 - 48*x202*x61 - x204*x60 - x26*x315 + x26*x316 - x28*x315 + x28*x316 - x315*x33 + x315 + x316*x33 - x316 + x318*x40 - x318*x51 + x320*x68 - x320*x70 - x321*x70 + x321*x73 + x60*x78 + 6 + x14*x2**8/s**8)
    sssv = t*x327*(x141*x325 - x191 + 7*x205 - x207 + x208 + x209 + x210 - 7*x211 - 7*x212 + x213 + x214 + x215 - 7*x216 + x222 + x223 - x224 - x225 - x226 + x322 + x323 + x324 + x326*x50 - x326*x59)
    ssst = x327*(x100*x328 - x103*x328 - x104*x328 - x110*x328 + x153*x325 + x172*x329 - x173*x329 - x191*x23 + x192 - x193 + 7*x200 + x221 + x23*x322 + x23*x323 + x23*x324 - x233 + x235 - x236 - x237 + x240 + x241 + x242 + x243 + x244 - x245 + x246 - x247 + x248 - x249 - x250 + x252 - x254 + x255 - x256 + x257 + x26*x326 + x28*x326 + x326*x33 - x326 - x328*x68)
    ssvv = 2*x2*x47*(x200 - x206 - x217 - x239 - x258 - x259 + x262 + x263 - x264 - x273 + x330)
    ssvt = x331*(-x170*x260 + x170*x261 - x190*x290 - x205 + x211 + x212 + x216 + x231 - x267 - x269 - x270 - x271 - x274 + x275 + x276 + x278 - x279 + x281 - x282)
    sstt = 2*x331*(-x180*x239 - x180*x273 + x180*x330 - x187 - x189 - x22*x228 - x22*x229 - x22*x230 - x280 + x283 - x284 - x285 + x286 + x287 - x288 - x289 - x291 - x293 + x294 + x295 + x296 + x297 - x298 + x299 - x300 + x312)
    svvv = t*x333*(x156 + x158 + x160 - x301 + 1)
    svvt = x333*(x304 + x334 + x50 - x59)
    svtt = x312*x332*(x272 + x310 - x311 + x334)
    sttt = x174*x333*(x108 + x313 - x314 + x334)
    vvvv = 2*x302
    vvvt = x22*x335
    vvtt = x174*x335
    vttt = x22**3*x335
    tttt = -2*x156*x336 - 2*x158*x336 - 2*x160*x336 + 2*x301*x336 - 2*x336 - 6/x11
         
    D4 = _expand_arrs(mmmm, mmms, mmmv, mmmt, mmss, mmsv, mmst, mmvv, mmvt, 
                      mmtt, msss, mssv, msst, msvv, msvt, mstt, mvvv, mvvt,
                      mvtt, mttt, ssss, sssv, ssst, ssvv, ssvt, sstt, svvv,
                      svvt, svtt, sttt, vvvv, vvvt, vvtt, vttt, tttt)
        
    return D4


def _pdf(x, m=0.0, s=1.0, v=0.0, t=1.0):
    return np.exp(_logpdf(x, m=m, s=s, v=v, t=t))


def _cdf(x, m=0.0, s=1.0, v=0.0, t=1.0):
    z = (x - m) / s
    r = np.sinh(t * np.arcsinh(z) - v)
    return sp.special.ndtr(r)


def _qtf(q, m=0.0, s=1.0, v=0.0, t=1.0):
    y = sp.special.ndtri(q)
    z = np.sinh((np.arcsinh(y) + v) / t)
    x = s * z + m
    return x

def _rvs(random_state, m=0.0, s=1.0, v=0.0, t=1.0, size=None):
    u = random_state.uniform(low=0.0, high=1.0, size=size)
    y = _qtf(u, m=m, s=s, v=v, t=t)
    return y

