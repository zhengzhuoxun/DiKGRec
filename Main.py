import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import DiKGRec
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import random

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
	
	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret
	
	def run(self):
		self.prepareModel()
		log('Model Prepared')
		log('Model Initialized')

		recallMax = 0
		ndcgMax = 0
		bestEpoch = 0

		for ep in range(0, args.epoch):
			if ep - bestEpoch >= 20:
				print('-'*18)
				print('Exiting from training early')
				break

			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax)


	def prepareModel(self):
		self.model = DiKGRec(args, self.handler.trnMat, self.handler.kg_mat_sparse_list).cuda()
		self.model_opt = torch.optim.Adam(self.model.parameters(), weight_decay=0)

		par_num1 = sum([param.nelement() for param in self.model.parameters()]) 
		print("Number of all parameters:", par_num1)

		

	def trainEpoch(self):
		self.model.train()
		epDfLoss, epKgLoss, epLoss = 0, 0, 0

		diffusionLoader = self.handler.diffusionLoader

		for i, (batch, index) in enumerate(diffusionLoader):
			batch, index = batch.cuda(), index.cuda()
			self.model_opt.zero_grad()

			kg_loss, diff_loss, total_loss = self.model.training_losses(batch)

			loss = total_loss.mean() 

			epKgLoss += kg_loss.mean().item()
			epDfLoss += diff_loss.mean().item()
			epLoss += total_loss.mean().item()

			loss.backward()
			self.model_opt.step()	

			log('Training Step %d/%d: batchLoss = %.4f, diffLoss = %.4f, kgLoss = %.4f' % 
	   (i, len(diffusionLoader), loss.item()*args.batch, diff_loss.mean().item()*args.batch, kg_loss.mean().item()*args.batch), 
				save=False, oneline=False)
		log('')

		ret = dict()
		ret['epLoss'] = epLoss 
		ret['epDfLoss'] = epDfLoss 
		ret['epKgLoss'] = epKgLoss 
		return ret
	
	def testEpoch(self):
		self.model.eval()		
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg = [0] * 2
		num = tstLoader.dataset.__len__()
		steps = len(tstLoader)

		with torch.no_grad():
			for i, (idx, usr, trnMask) in enumerate(tstLoader):
				trnMask = trnMask.float().cuda()
				prediction_batch = self.model.inference(trnMask).cpu()
				
				prediction_batch = prediction_batch - trnMask.cpu() * 1e8
				_, topLocs = t.topk(prediction_batch, args.topk)
				recall, ndcg = self.calcRes(topLocs.numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
				epRecall += recall
				epNdcg += ndcg
				log('Steps %d/%d: batch_recall = %.2f, batch_ndcg = %.2f ' % (i, steps, recall, ndcg), save=False, oneline=True)

		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg
	
def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')
	log(args)

	coach = Coach(handler)
	coach.run()

