import numpy as np

# Repeat 노드의 역전파
D, N = 8, 7
x = np.random.randn(1, D)   # 입력
y = np.repeat(x, N, axis=0)     # 순전파(벡터 x를 N번 복제하는데, axis를 지정하여 어느 축 방향으로 복제할지 조정함)

dy = np.random.randn(N, D)  # 무작위 기울기
dx = np.sum(dy, axis=0, keepdims=True)  # 역전파(keepdims : True면 2차원 배열의 차원 수 유지, False면 벡터로 반환됨)
print(x.shape, y.shape, dy.shape, dx.shape)

# Sum 노드의 역전파
a = np.random.randn(N, D)   # 입력
b = np.sum(a, axis=0, keepdims=True)    # 순전파

db = np.random.randn(1, D)  # 무작위 기울기
da = np.repeat(db, N, axis=0)   # 역전파
print(a.shape, b.shape, db.shape, da.shape)

# MatMul 노드
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW     # ... : 생략기호(ellipsis). 변수의 메모리 주소 고정하는 깊은 복사 실행
        return dx
