import torch
from torch import nn
import numpy as np
from torch.autograd import gradcheck, Function


class TanhFixedPoint(Function):
	"""
	Find the fixed point of z = tanh(Wz + x)
	W: [n, n]
	x: [batch, n]
	z: [batch, n]
	"""
	@staticmethod
	def forward(ctx, x, W, max_iter=500, eps=1e-5):
		"""
		Solve the fixed point of z = tanh(Wz + x) with Newton's method
		Remark: Empirically, W should be initialized with kaiming_uniform_, otherwise it may converge much slower
		:param ctx:
		:param x: [batch, n]
		:param W: [n, n]
		:param max_iter: int
		:param eps: float
		:return:
		"""
		with torch.no_grad():
			z = torch.zeros_like(x)
			iterations = 0
			for _ in range(max_iter):
				iterations += 1

				tanh = torch.tanh(z @ W.T + x)
				g = z - tanh # [batch, n]

				tanhp = 1 - tanh ** 2 # [batch, n]
				I = torch.eye(W.shape[0]).to(W.device)[None, :, :] # [1, n, n]
				D = I - tanhp[:, :, None] * W[None, :, :] # [batch, n, n]
				dz = torch.linalg.solve(D, -g[:, :, None])[:, :, 0] # [batch, n]

				if torch.norm(g, p=torch.inf) < eps:
					break

				z +=  dz
		ctx.save_for_backward(z, D, tanhp)
		return z

	@staticmethod
	def backward(ctx, dl_dz):

		"""
		:param ctx:
		:param dl_dz: [batch, n]
		:return:
		"""

		z, D, tanhp = ctx.saved_tensors # [batch, n], [batch, n, n], [batch, n]
		temp = torch.linalg.solve(D.permute(0,2,1), dl_dz[:, :, None])[:, :, 0] # [batch, n]

		# gradient wrt input x
		dl_dx = tanhp * temp # [batch, n]

		# gradient wrt weight W
		temp2 = torch.bmm(tanhp[:, :, None], z[:, None, :]) # [batch, n, n]
		dl_dW =  temp[:, :, None] * temp2 # [batch, n, n]
		dl_dW = torch.sum(dl_dW, dim=0) # [n, n]

		return dl_dx, dl_dW

class TanhFixedPointLayer(nn.Module):
	def __init__(self, dim):
		super(TanhFixedPointLayer, self).__init__()
		W = torch.zeros(dim, dim)
		nn.init.kaiming_uniform_(W, a=np.sqrt(5))
		self.W = nn.Parameter(W, requires_grad=True)

	def forward(self, x):
		return TanhFixedPoint.apply(x, self.W)

def gradeint_check_fixed_point():
	n = 50
	batch = 3
	W = nn.Parameter(torch.randn(n, n), requires_grad=True).double()
	nn.init.kaiming_uniform_(W, a=np.sqrt(5))
	x = nn.Parameter(torch.randn(batch, n), requires_grad=True).double()

	gradcheck(TanhFixedPoint.apply, (x, W), eps=1e-4, atol=1e-4, rtol=1e-4)

def MNIST():
	"""
	Most of this function is generated by CoPilot
	Hail GPT!
	"""
	from torchvision import datasets, transforms
	from torch.utils.data import DataLoader
	from torch import optim
	from torch.nn import functional as F
	from sklearn.metrics import accuracy_score

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
	                          							  batch_size=128, shuffle=True)
	test_loader = DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
	                          							  batch_size=128, shuffle=True)

	model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(28*28, 100),
		nn.Tanh(),
		TanhFixedPointLayer(100),
		nn.Linear(100, 10)
	).to(device)

	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = F.cross_entropy

	for e in range(2):
		for b, (x, y) in enumerate(train_loader):
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			y_hat = model(x)
			loss = loss_fn(y_hat, y)
			loss.backward()
			optimizer.step()

			yhat = torch.argmax(y_hat, dim=1)
			acc = accuracy_score(y.cpu().numpy(), yhat.cpu().numpy())
			print(f'Epoch {e}, batch {b} / {len(train_loader)}, loss {loss.item():.4f}, acc {acc:.4f}')





if __name__ == '__main__':
	gradeint_check_fixed_point()

	# MNIST()





