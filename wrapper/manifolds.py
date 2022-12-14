import numpy as np
import torch as th


class Euclidean():
    "Class for Euclidean manifolds."

    def __init__(self, *shape):
        if len(shape) == 0:
            raise TypeError("Need shape parameters")
        if len(shape) == 1:
            name = "Euclidean manifold of {}-vectors".format(*shape)
        elif len(shape) == 2:
            name = ("Euclidean manifold of {}x{} matrices").format(*shape)
        else:
            name = ("Euclidean manifold of shape " + str(shape) + " tensors")
        #dimension = np.prod(shape)
        self._shape = shape

    def inner(self, X, G, H):
        return float(th.tensordot(G, H, dims=G.dim()))

    def norm(self, X, G):
        return th.linalg.norm(G)

    def dist(self, X, Y):
        return th.linalg.norm(X - Y)

    def proj(self, X, U):
        return U

    def exp(self, X, U):
        return X + U

    def log(self, X, Y):
        return Y - X

    def rand(self, dim):
        #return th.random.randn(*self._shape)
        return th.randn(dim)

    def transp(self, X1, X2, G):
        return G


class Sphere():
    "Class for sphere manifolds."

    def __init__(self, *shape):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        if len(shape) == 1:
            name = "Sphere manifold of {}-vectors".format(*shape)
        else:
            raise NotImplementedError
        #dimension = th.prod(th.tensor(shape)) - 1
        self._shape = shape

    def inner(self, X, U, V):
        return th.tensordot(U, V, dims=U.dim())

    def norm(self, X, U):
        return th.linalg.norm(U)

    def dist(self, U, V):
        # Make sure inner product is between -1 and 1
        inner = self.inner(None, U, V).clamp(-1, 1)
        # inner = th.max(th.min(self.inner(None, U, V), th.tensor[1]), [-1])
        return th.arccos(inner)

    def proj(self, X, H):
        return H - self.inner(None, X, H) * X

    def exp(self, X, U):
        norm_U = self.norm(None, U)
        # Check that norm_U isn't too tiny. If very small then
        # sin(norm_U) / norm_U ~= 1 and retr is extremely close to exp.
        if norm_U > 1e-3:
            return X * th.cos(norm_U) + U * th.sin(norm_U) / norm_U
        else:
            return self.normalize(X + U)

    def log(self, X, Y):
        P = self.proj(X, Y - X)
        #dist = self.dist(X, Y)
        # If the two points are "far apart", correct the norm.
        #if dist > 1e-6:
        #    P *= dist / self.norm(None, P)
        P = P * self.norm(None, P)
        return P

    def rand(self, dim):
        #Y = th.random.randn(*self._shape)
        Y = th.randn(dim)
        return self.normalize(Y)

    def transp(self, X, Y, U):
        # return self.proj(Y, U)  # approximation
        x_dir = self.log(X, Y)
        norm_x_dir = th.linalg.norm(x_dir)
        if norm_x_dir > 1e-3:
          normalized_x_dir = x_dir / norm_x_dir
          trsp_operator = th.tensordot(-X * th.sin(norm_x_dir), normalized_x_dir, axes=0) + \
                          th.tensordot(normalized_x_dir * th.cos(norm_x_dir), normalized_x_dir, axes=0) + \
                          th.eye(x_dir.shape[0]) - th.tensordot(normalized_x_dir, normalized_x_dir, axes=0)
          return th.dot(trsp_operator, U)
        else:
          return self.proj(Y, U)

    def normalize(self, X):
        return X / self.norm(None, X)


class SphereNP():
    "Class for sphere manifolds."

    def __init__(self, *shape):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        if len(shape) == 1:
            name = "Sphere manifold of {}-vectors".format(*shape)
        else:
            raise NotImplementedError
        dimension = np.prod(shape) - 1
        self._shape = shape

    def inner(self, X, U, V):
        return float(np.tensordot(U, V, axes=U.ndim))

    def norm(self, X, U):
        return np.linalg.norm(U)

    def dist(self, U, V):
        # Make sure inner product is between -1 and 1
        inner = max(min(self.inner(None, U, V), 1), -1)
        return np.arccos(inner)

    def proj(self, X, H):
        return H - self.inner(None, X, H) * X

    def exp(self, X, U):
        # TODO COMPLETE
        norm_U = self.norm(None, U)
        # Check that norm_U isn't too tiny. If very small then
        # sin(norm_U) / norm_U ~= 1 and retr is extremely close to exp.
        if norm_U > 1e-3:
            return X * np.cos(norm_U) + U * np.sin(norm_U) / norm_U
        else:
            return self.normalize(X + U)

    def log(self, X, Y):
        # TODO COMPLETE
        P = self.proj(X, Y - X)
        dist = self.dist(X, Y)
        # If the two points are "far apart", correct the norm.
        if dist > 1e-6:
            P *= dist / self.norm(None, P)
        return P

    def rand(self):
        Y = np.random.randn(*self._shape)
        return self.normalize(Y)

    def transp(self, X, Y, U):
        # return self.proj(Y, U)  # approximation
        x_dir = self.log(X, Y)
        norm_x_dir = np.linalg.norm(x_dir)
        if norm_x_dir > 1e-3:
          normalized_x_dir = x_dir / norm_x_dir
          trsp_operator = np.tensordot(-X * np.sin(norm_x_dir), normalized_x_dir, axes=0) + \
                          np.tensordot(normalized_x_dir * np.cos(norm_x_dir), normalized_x_dir, axes=0) + \
                          np.eye(x_dir.shape[0]) - np.tensordot(normalized_x_dir, normalized_x_dir, axes=0)
          return np.dot(trsp_operator, U)
        else:
          return self.proj(Y, U)

    def normalize(self, X):
        return X / self.norm(None, X)

"""
U = th.tensor([[1,0]])
X = th.tensor([[1,0], [0,1], [1/2, th.sqrt(th.tensor([3]))/2]]).to(th.float)
#X = th.tensor([[1,1]])
M = Sphere(3)
a = M.exp(U, X)
print(U)
print(a)


U = np.sqrt([[1/2,1/2]])
#X = th.tensor([[1,0], [0,1], [1/2, th.sqrt(th.tensor([3]))/2]])
#X = np.sqrt([[1,0], [0,1], [1/2, th.sqrt(th.tensor([3]))/2]])
#X = np.sqrt([[1,1]])
_, _, b = np.linalg.svd(np.sqrt([[1/2, 1/2]]))

print("b", b)
X = np.array([[0.8, 1], [1,0.7], [1/3, np.sqrt([3])/2]]) * b[1]
M = SphereNP(3)
a = M.exp(U, X)
print(X)
print(a)
"""

