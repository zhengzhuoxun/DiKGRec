import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Utils.TimeLogger import log
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random
import math

class DataHandler:
	def __init__(self):
		if args.data == 'lastfm':
			predir = 'Datasets/lastfm/'
		elif args.data == 'amazon-book':
			predir = 'Datasets/amazon-book/'
		elif args.data == 'yelp2018':
			predir = 'Datasets/yelp2018/'
		self.predir = predir
		self.trnfile = predir + 'train.txt'
		self.tstfile = predir + 'test.txt'
		self.kgfile = predir + 'kg.txt'
	
	def read_cf(self, file_name):
		inter_mat = list()
		lines = open(file_name, "r").readlines()
		for l in lines:
			tmps = l.strip()
			inters = [int(i) for i in tmps.split(" ")]

			u_id, pos_ids = inters[0], inters[1:]
			pos_ids = list(set(pos_ids))
			for i_id in pos_ids:
				inter_mat.append([u_id, i_id])
		return np.array(inter_mat)
	
	def remap_item(self, train_data, test_data):
		args.user = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
		args.item = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1
		trn_Mat = sp.coo_matrix((np.ones_like(train_data[:,0]), 
						   (train_data[:,0], train_data[:,1])), dtype='float32', shape=(args.user, args.item))
		tst_Mat = sp.coo_matrix((np.ones_like(test_data[:,0]), 
						   (test_data[:,0], test_data[:,1])), dtype='float32', shape=(args.user, args.item))
		return trn_Mat, tst_Mat

	def readTriplets(self, file_name):
		can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
		can_triplets_np = np.unique(can_triplets_np, axis=0)

		inv_triplets_np = can_triplets_np.copy()
		inv_triplets_np[:, 0] = can_triplets_np[:, 2]
		inv_triplets_np[:, 2] = can_triplets_np[:, 0]
		inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
		triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

		n_relations = max(triplets[:, 1]) + 1

		args.relation = n_relations

		args.entity = max(max(triplets[:, 0]), max(triplets[:, 1])) + 1

		return triplets
	
	def buildGraphs(self, triplets):
		print("Begin to load knowledge graph triples ...")
		indices_all = defaultdict(list)
		values_all = defaultdict(list)
		rel_set = set()
		kg_mat_sparse_list = []

		kg_dict_tail = defaultdict(list)
		kg_dict_head = defaultdict(list)
		for h_id, r_id, t_id in tqdm(triplets, ascii=True):
			kg_dict_tail[(h_id, r_id)].append(t_id)
			kg_dict_head[(t_id, r_id)].append(h_id)
		
		for h_id, r_id, t_id in tqdm(triplets, ascii=True):
			tailNum = len(kg_dict_tail[(h_id, r_id)])
			headNum = len(kg_dict_head[(t_id, r_id)])
			rel_set.add(r_id)
			indices_all[r_id].append([h_id, t_id])
			if args.kg_norm == 0:
				values_all[r_id].append(1)
			elif args.kg_norm == 1:
				values_all[r_id].append(1 / tailNum)
			elif args.kg_norm == 2:
				values_all[r_id].append(1 / (math.sqrt(tailNum * headNum)))

		for rel in rel_set:
			indices_r = torch.tensor(indices_all[rel], dtype=torch.long).t()  # [2, tripleNum]
			values_r = torch.tensor(values_all[rel], dtype=torch.float)	# [tripleNum]
			kg_mat_sparse_r = torch.sparse_coo_tensor(indices_r, values_r, 
                                            size=(args.entity, args.entity))
			
			kg_mat_sparse_r = kg_mat_sparse_r.to_sparse_csr()
			kg_mat_sparse_list.append(kg_mat_sparse_r)
		return kg_mat_sparse_list
	
	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	
	def find_cold_start_id(self, mat):
		csc_matrix = mat.tocsc()
		nonzero_columns = np.where(np.diff(csc_matrix.indptr) > 0)[0]
		selected_cols = np.random.choice(nonzero_columns, size=min(args.cold_start_num, len(nonzero_columns)), replace=False)

		return selected_cols

	def LoadData(self):
		trn_cf = self.read_cf(self.trnfile)
		tst_cf = self.read_cf(self.tstfile)
		trnMat, tstMat = self.remap_item(trn_cf, tst_cf)

		if args.cold_start_num != 0:
			cold_start_ids = self.find_cold_start_id(tstMat)
			lil_matrix = trnMat.tolil()
			for col in cold_start_ids:
				lil_matrix[:, col] = 0
			trnMat = lil_matrix.tocoo()		
		
	
		self.trnMat = trnMat
		self.tstMat = tstMat
		self.torchBiAdj = self.makeTorchAdj(trnMat)

		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		kg_triplets = self.readTriplets(self.kgfile)
		self.kg_mat_sparse_list = self.buildGraphs(kg_triplets) # list of R sparse tensor, shape E*E
		print("In KG: %d relations, %d entities, %d triples" % (args.relation, args.entity, len(kg_triplets)))

		self.diffusionData = DiffusionData(torch.FloatTensor(trnMat.toarray()))
		self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, pin_memory=True, shuffle=True, num_workers=1)
		

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return idx, self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
		

class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item, index
    def __len__(self):
        return len(self.data)