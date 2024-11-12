import numpy as np

def hypot2(x, y):
    """Calculate the hypotenuse for x and y to avoid overflow/underflow."""
    return np.sqrt(x*x + y*y)

def tred2(V):
    """Householder reduction of symmetric matrix to tridiagonal form."""
    n = V.shape[0]
    d = np.zeros(n)
    e = np.zeros(n)
    
    for j in range(n):
        d[j] = V[n-1, j]

    for i in range(n-1, 0, -1):
        scale = np.sum(np.abs(d[:i]))
        if scale == 0:
            e[i] = d[i-1]
            d[:i] = V[i-1, :i]
            V[i, :i] = 0.0
            V[:i, i] = 0.0
        else:
            d[:i] /= scale
            h = np.dot(d[:i], d[:i])
            f = d[i-1]
            g = -np.sqrt(h) if f > 0 else np.sqrt(h)
            e[i] = scale * g
            h -= f * g
            d[i-1] = f - g
            e[:i] = 0.0
            
            for j in range(i):
                V[j, i] = d[j]
                g = np.dot(V[j, :j+1], d[:j+1]) + np.dot(V[j+1:i, j], d[j+1:i])
                e[j] += V[j, j] * d[j]
            e[:i] /= h
            f = np.dot(e[:i], d[:i])
            hh = f / (h + h)
            e[:i] -= hh * d[:i]
            for j in range(i):
                f = d[j]
                g = e[j]
                V[:j+1, j] -= f * e[:j+1] + g * d[:j+1]
            d[:i] = V[i-1, :i]
            V[i, :i] = 0.0

        d[i] = h

    for i in range(n-1):
        V[n-1, i] = V[i, i]
        V[i, i] = 1.0
        h = d[i+1]
        if h != 0:
            d[:i+1] = V[:i+1, i+1] / h
            for j in range(i+1):
                g = np.dot(V[:i+1, i+1], V[:i+1, j])
                V[:i+1, j] -= g * d[:i+1]
        V[:i+1, i+1] = 0.0
    V[n-1, n-1] = 1.0
    e[0] = 0.0

    return V, d, e

def tql2(V, d, e):
    """QL algorithm with implicit shifts, finding eigenvalues and eigenvectors."""
    n = len(d)
    e[:-1] = e[1:]
    e[-1] = 0.0
    f = 0.0
    tst1 = 0.0
    eps = 2.0**-52.0
    
    for l in range(n):
        tst1 = max(tst1, abs(d[l]) + abs(e[l]))
        m = l
        while m < n:
            if abs(e[m]) <= eps * tst1:
                break
            m += 1
        if m > l:
            iter_count = 0
            while True:
                iter_count += 1
                g = d[l]
                p = (d[l+1] - g) / (2.0 * e[l])
                r = hypot2(p, 1.0)
                if p < 0:
                    r = -r
                d[l] = e[l] / (p + r)
                h = g - d[l]
                d[l+1] -= h
                f += h
                for i in range(l+2, n):
                    d[i] -= h
                p = d[m]
                c = 1.0
                s = 0.0
                for i in range(m-1, l-1, -1):
                    g = c * e[i]
                    h = c * p
                    r = hypot2(p, e[i])
                    e[i+1] = s * r
                    s = e[i] / r
                    c = p / r
                    p = c * d[i] - s * g
                    d[i+1] = h + s * (c * g + s * d[i])
                    for k in range(n):
                        h = V[k, i+1]
                        V[k, i+1] = s * V[k, i] + c * h
                        V[k, i] = c * V[k, i] - s * h
                e[l] = s * p
                d[l] = c * p
                if abs(e[l]) <= eps * tst1:
                    break
        d[l] += f
        e[l] = 0.0

    idx = d.argsort()
    d[:] = d[idx]
    V[:] = V[:, idx]
    return V, d

def eigen_decomposition(A):
    """Compute eigenvalues and eigenvectors of a symmetric 3x3 matrix."""
    V = np.array(A, dtype=float)
    V, d, e = tred2(V)
    V, d = tql2(V, d, e)
    return V, d

# Example usage
A = np.array([[4, 1, 1],
              [1, 3, 1],
              [1, 1, 2]])
V, d = eigen_decomposition(A)
print("Eigenvalues:", d)
print("Eigenvectors:\n", V)
