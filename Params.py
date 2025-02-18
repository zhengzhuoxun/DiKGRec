import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
	parser.add_argument('--lr2', default=5e-5, type=float, help='learning rate')
	parser.add_argument('--batch', default=400, type=int, help='batch size')
	parser.add_argument('--tstBat', default=512, type=int, help='number of users in a testing batch')
	parser.add_argument('--epoch', default=1000, type=int, help='number of epochs')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--data', default='yelp2018', type=str, help='name of dataset')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
	parser.add_argument("--seed", type=int, default=421, help="random seed")

	parser.add_argument('--dims', type=str, default='[1000]')
	parser.add_argument('--d_emb_size', type=int, default=10)
	parser.add_argument('--norm', type=bool, default=True)
	parser.add_argument('--steps', type=int, default=5)
	parser.add_argument('--noise_scale', type=float, default=5e-4)
	parser.add_argument('--noise_min', type=float, default=0.0005)
	parser.add_argument('--noise_max', type=float, default=0.005)
	parser.add_argument('--sampling_steps', type=int, default=0)
	parser.add_argument('--sampling_N', type=int, default=20)

	parser.add_argument('--head', type=int, default=2)

	parser.add_argument('--res_lambda', type=float, default=0.5)

	parser.add_argument('--kg_norm', default=2, type=int)
	parser.add_argument('--layer', default=4, type=int)
	parser.add_argument('--updateW', default=1, type=float, help='in KG aggregator')
	parser.add_argument('--oriW', default=0, type=float, help='in KG aggregator')
	parser.add_argument('--noise_ratio', default=0.2, type=float)
	parser.add_argument('--kg_loss_ratio', default=0.4, type=float)
	parser.add_argument('--head_diff_ratio', default=0, type=float)

	parser.add_argument('--cold_start_num', default=0, type=int)

	parser.add_argument('--diff_type', type=int, default=0)
	parser.add_argument('--trans_ratio', default=0, type=float)

	return parser.parse_args()
args = ParseArgs()