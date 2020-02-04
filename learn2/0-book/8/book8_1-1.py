# coding=utf-8

import numpy as np

def reweight_distribution(original_distribution, temperature=0.5):
	distribution = np.log(original_distribution)/temperature	#  np.log() 以e为底的对数运算,a=np.log(b)，e^a=b
	# print(np.log(original_distribution))	# e^x=original_distribution 求x的值，相当于求多少个e想成得到original_distribution

	distribution = np.exp(distribution)		# np.exp() e的多少次方运算
	# print(distribution)

	return distribution / np.sum(distribution)

if __name__ == '__main__':
	original_distribution = 2.71
	# original_distribution 	# 是概率值组成的一维Numpy数组，这些概率值之 和必须等于 1。temperature 是一个 因子，用于定量描述输出分布的熵
	print(reweight_distribution(original_distribution))
	