import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
from torch_scatter import scatter_sum, scatter_softmax
import math
from scipy.sparse import coo_matrix
import time

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DiKGRec(nn.Module):
	def __init__(self, args, uiMat, kgMatSparseList, beta_fixed=True, dropout=0.5):
		# uiMat in sp.coo_matrix
		super(DiKGRec, self).__init__()
		self.entity = args.entity
		self.user = args.user
		self.item = args.item
		self.head = args.head
		self.relation = args.relation
		self.kg_weight = args.kg_loss_ratio
		self.trans_ratio = args.trans_ratio
		self.sampling_N = args.sampling_N

		self.usrEmb = torch.empty(self.user, self.head).cuda()

		self.buildAggregator(uiMat, kgMatSparseList)
		self.buildDiffusion(args, beta_fixed)
		self.buildDenoiser(eval(args.dims), dropout)

		self.param_groups = [
            {'params': (
                    list(self.emb_layer.parameters()) +
                    [p for layer in self.in_layers for p in layer.parameters()] +
                    [p for layer in self.out_layers for p in layer.parameters()]
                ), 'lr': args.lr, 'weight_decay': 0},
            {'params': (
                    [self.relEmb]
                ), 'lr': args.lr2, 'weight_decay': 0}
        ]


	def buildAggregator(self, uiMat, kgMatSparseList):
		self.relEmb = nn.Parameter(torch.randn(self.head, len(kgMatSparseList)))
		self.kgMatList = kgMatSparseList 
		self.layer = args.layer
		self.updateW = args.updateW
		self.oriW = args.oriW
	
	def buildDiffusion(self, args, beta_fixed):
		self.noise_scale = args.noise_scale
		self.noise_min = args.noise_min
		self.noise_max = args.noise_max
		self.steps = args.steps

		if args.noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001
			self.calculate_for_diffusion()

	
	def buildDenoiser(self, hidden_dims, dropout):
		self.in_dims = [self.item] + hidden_dims
		self.out_dims = hidden_dims + [self.item]
		self.time_emb_dim = args.d_emb_size
		self.norm = args.norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)

		self.init_weights()
	
	
	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)


	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas)
	
	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, x_start, steps):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)

		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(x_t, t)
			x_t = model_mean	

		# x_t = self.kgAggregation(x_t)
				
		return x_t

			
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
	
	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)
	
	def p_mean_variance(self, x, t):
		model_output = self.denoise(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output 
				+ self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance
	
	def denoise(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)
		# h = self.kgAggregation(h)
		return h
	
	def kgAggregation(self, batchUI, x_start, batch_index, mode=0):
		uBatchSize= batchUI.shape[0]
		# with torch.no_grad():
		entEmb = torch.zeros(self.entity, uBatchSize).cuda()

		# entEmb[self.item:, :] = self.entEmb[:, batch_index]
		entEmb[:self.item, :] = batchUI.T 

		error_heads = torch.empty(uBatchSize, self.head).cuda() 
		usr_pred = torch.empty(uBatchSize, self.item, self.head).cuda()  
		for h in range(self.head):
			entEmb_h = entEmb.clone()
			relEmb = self.relEmb[h,:]
			for i in range(self.layer):
				neighbor_weighted_sum = torch.zeros(self.entity, uBatchSize).cuda()
				normalized_relEmb = torch.softmax(relEmb, dim=0).cuda()
				for i, kgMat in enumerate(self.kgMatList):
					neighbor_sum_r = torch.sparse.mm(kgMat.cuda(), entEmb_h) # [E, E] * [E, uB]
					neighbor_weighted_sum += neighbor_sum_r * normalized_relEmb[i]
				entEmb_h = entEmb_h* self.oriW + neighbor_weighted_sum * self.updateW
			itEmb_h = entEmb_h[:self.item, :] # [item, uBatchSize]
			usr_pred_h = itEmb_h.T # [uBatchSize, item]

			if mode==0:
				error_h = self.error(usr_pred_h, x_start)
				error_heads[:, h] = error_h
			usr_pred[:,:,h] = usr_pred_h

		if mode == 0:
			self.usrEmb[batch_index, :] = F.softmax(error_heads.detach(), dim=1)
		
		weights = self.usrEmb[batch_index, :].unsqueeze(1)
		weighted_usr_pred = usr_pred * weights
		final_output = weighted_usr_pred.sum(dim=2)
		
		return final_output
	
	def error(self, x_start, out):
		mse = self.mean_flat((out - x_start) ** 2)
		return mse
	

	def noise_filter(self, mat, ratio = 0.2):
		with torch.no_grad():
			filteredMat = mat.clone()
			ones_indices = torch.nonzero(mat == 1, as_tuple=False)
			num_ones = ones_indices.size(0)
			to_filter = int(num_ones * ratio)
			indices_to_zero = ones_indices[torch.randperm(num_ones)[:to_filter]]
			filteredMat[indices_to_zero[:, 0], indices_to_zero[:, 1]] = 0
		return filteredMat.detach(), indices_to_zero.detach()
	
	def bernoulli_filter(self, mat, ratio=0.2, N=10):
		with torch.no_grad():
			results = []
			for i in range(N):
				torch.manual_seed(torch.seed() + i) 
				mask = (torch.rand_like(mat, dtype=torch.float) < ratio).int().cuda()  
				filteredMat = mat * (1 - mask)
				results.append(filteredMat.detach())
		return results, mask

	def training_losses(self, x_start, batch_index):
		# mat, indices_to_zero = self.noise_filter(x_start, args.noise_ratio)
		batch_size = x_start.size(0)
		mat_list, indices_to_zero = self.bernoulli_filter(x_start, args.noise_ratio, self.sampling_N)

		kg_in = mat_list[0]
		di_in = mat_list[1]

		if self.kg_weight != 0:
			kgPred = self.kgAggregation(mat_list[0], x_start, batch_index)
			
			kg_loss = self.error(x_start, kgPred)

			head_diff = self.head_diff()

			kg_loss += args.head_diff_ratio * head_diff 
		else:
			kg_loss = torch.zeros(batch_size).cuda()

		
		noise = torch.randn_like(x_start)

		if args.diff_type == 0:
			diff_input = x_start
		else:
			diff_input = mat_list[1]
		
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()

		if self.noise_scale != 0:
			x_t = self.q_sample(diff_input, ts, noise)
		else:
			x_t = diff_input

		diffPred = self.denoise(x_t, ts)

		weight = self.SNR(ts - 1) - self.SNR(ts)
		
		diff_mse = self.mean_flat((diff_input - diffPred) ** 2)

		
		weight = torch.where((ts == 0), 1.0, weight)
		diff_loss = weight * diff_mse 

		trans_loss = torch.zeros(batch_size).cuda()

		if args.diff_type != 0:
			stacked_mats = torch.stack(mat_list, dim=0) # [N, batch, item]
			trans_diff = ((stacked_mats - x_start) ** 2).sum(dim=2) # [N, batch]
			trans_loss = trans_diff.mean(dim=0)  # [batch]
			trans_loss = trans_loss * self.trans_ratio

		loss = self.kg_weight * kg_loss + (1-self.kg_weight) * (diff_loss + trans_loss)
		return kg_loss, diff_loss, trans_loss, loss
	
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
	

	def inference(self, mat, batch_index): 
		kgOut = self.kgAggregation(mat, mat, batch_index, 1)
		diffOut = self.p_sample(mat, args.sampling_steps)

		out = self.kg_weight * kgOut + (1-self.kg_weight) * diffOut

		return out
	
	def head_diff(self):
		mean_emb = torch.mean(self.relEmb, dim=0)
		variance_loss = - torch.sum((self.relEmb - mean_emb) ** 2)

		return variance_loss
	