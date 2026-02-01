from __future__ import annotations
import numpy as np
from typing import List, Any
from typing import List, Sequence

Array = np.ndarray

def compress(mpo: List[Any],
             D_max: int,
             svd_eps: float = 1e-30) -> List[Any]:
    
    L  = len(mpo)
    d  = mpo[0].shape[1]
    p  = d * d                   
    C  = [T.copy() for T in mpo] 

    #RIGHT-TO-LEFT  SVD  SWEEP
   
    for i in range(L - 1, 0, -1):
        Dl, _, _, Dr = C[i].shape
        M = C[i].reshape(Dl, p * Dr)                 # (D_L, p*D_R)

        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        r_s = min(D_max, np.count_nonzero(S > svd_eps))
        r_s = max(r_s, 1) 
        U, S, Vh = U[:, :r_s], S[:r_s], Vh[:r_s, :]
        C[i] = Vh.reshape(r_s, d, d, Dr)
        if not np.all(np.isfinite(S)):
            print("Non-finite S detected:", S)

        US = U @ np.diag(S)                          # (D_L, r_s)

        Dl1, _, _, Dr1 = C[i - 1].shape
        assert Dr1 == Dl, f"bond mismatch between sites {i-1} and {i}"

        left = C[i - 1].reshape(Dl1 * p, Dl) @ US    # (D_L1*p, r_s)
        C[i - 1] = left.reshape(Dl1, d, d, r_s)

    
    # LEFT-TO-RIGHT  QR  SWEEP
    for i in range(0, L - 1):
        Dl, _, _, Dr = C[i].shape             # (D_L, d, d, D_R)
        M = C[i].reshape(Dl * p, Dr)          # (D_L*p, D_R)

        Q, R = np.linalg.qr(M, mode='reduced')  # Q: (D_L*p, r_q),  R: (r_q, D_R)
        r_q  = min(D_max, Q.shape[1])
        Q, R = Q[:, :r_q], R[:r_q, :]          
        C[i] = Q.reshape(Dl, d, d, r_q)
        Dl1, _, _, Dr1 = C[i + 1].shape
        assert Dl1 == Dr, f"bond mismatch between sites {i} and {i+1}"

        right = R @ C[i + 1].reshape(Dl1, p * Dr1)   # (r_q, p*D_R1)
        C[i + 1] = right.reshape(r_q, d, d, Dr1)

    

    return C

def product(WA: List[Array], WB: List[Array]) -> List[Array]:
   
    L = len(WA)
    C  = []

    for i in range(L):
        A = WA[i]               
        B = WB[i]                   

        if A.shape[1:3] != B.shape[1:3]:
            raise ValueError(f"Physical dims differ at site {i}")
        d = A.shape[1]

        DₐL, _, _, DₐR = A.shape
        DᵦL, _, _, DᵦR = B.shape

        tmp = np.tensordot(A, B, axes=([2], [1]))

        tmp = tmp.transpose(0, 3, 1, 4, 2, 5)

        C_i = tmp.reshape(DₐL * DᵦL, d, d, DₐR * DᵦR)

        C.append(C_i)

    return C


def trace(Wlist):
    
    Ti = []
    for W in Wlist:
        D_left, d, _, D_right = W.shape
        T  = np.zeros((D_left, D_right), dtype=W.dtype)
        for σ in range(d):
            T += W[:, σ, σ, :]
        Ti.append(T)

    M = Ti[0] 
    for T in Ti[1:]:
        M = M @ T   
    return M.item()

I  = np.eye(2, dtype=complex)
S_plus = np.array([[0, 1], [0, 0]], dtype=complex)
S_minus = np.array([[0, 0], [1, 0]], dtype=complex)
Sx = (S_plus + S_minus) / np.sqrt(2)
Sy = (S_plus - S_minus) / (1j * np.sqrt(2))
zero=np.zeros(2)


def identity_mpo(L: int) -> List[Array]:
    return [I.reshape(1, 2, 2, 1).copy() for _ in range(L)]

def xy_hamiltonian_mpo(L: int, J: float = 1.0) -> List[Array]:
    
    D = 4
    W: List[Array] = [None] * L
    Wl = np.zeros((1, 2, 2, D), dtype=complex)
    Wl[0, :, :, 0] = 0
    Wl[0, :, :, 1] = J*Sx
    Wl[0, :, :, 2] = J*Sy
    Wl[0, :, :, 3] = I
    W[0] = Wl

    Wr = np.zeros((D, 2, 2, 1), dtype=complex)
    Wr[0, :, :, 0] = I
    Wr[1, :, :, 0] = Sx
    Wr[2, :, :, 0] = Sy
    Wr[3, :, :, 0] = 0
    W[-1] = Wr

    Wb = np.zeros((D, 2, 2, D), dtype=complex)
    Wb[0, :, :, 0] = I
    Wb[1, :, :, 0] = Sx
    Wb[2, :, :, 0] = Sy
    Wb[3, :, :, 1] = J * Sx
    Wb[3, :, :, 2] =  J * Sy
    Wb[3, :, :, 3] = I
    for i in range(1, L - 1):
        W[i] = Wb.copy()

    return W
def direct_sum(A: Sequence[Array], B: Sequence[Array]) -> List[Array]:
    L = len(A)
    C: List[Array] = [None] * L

    for i in range(L):
        DlA, d, _, DrA = A[i].shape
        DlB, _, _, DrB = B[i].shape

        if DlA == DlB == 1:
            DlC, DrC = 1, DrA + DrB
            C[i] = np.zeros((1, d, d, DrC), dtype=complex)
            C[i][0, :, :, :DrA]      = A[i][0]
            C[i][0, :, :, DrA:DrC]   = B[i][0]

      
        elif DrA == DrB == 1:
            DlC, DrC = DlA + DlB, 1
            C[i] = np.zeros((DlC, d, d, 1), dtype=complex)
            C[i][:DlA, :, :, 0]      = A[i][:, :, :, 0]
            C[i][DlA:DlC, :, :, 0]   = B[i][:, :, :, 0]

        else:
            DlC, DrC = DlA + DlB, DrA + DrB
            C[i] = np.zeros((DlC, d, d, DrC), dtype=complex)
            C[i][:DlA, :, :, :DrA]   = A[i]
            C[i][DlA:DlC, :, :, DrA:DrC] = B[i]

    return C
    
def mpo(L: int, beta: float, J: float = 1.0) -> List[Array]:
    Id = identity_mpo(L)
    H  = xy_hamiltonian_mpo(L, J)
    k=(beta)**(1/L)
    for i in range(L):
        H[i] = (-k) * H[i]

    return direct_sum(Id, H)

