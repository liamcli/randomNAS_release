import pickle
import sys
import ast


def parse_arch_to_darts(benchmark, arch):
    op_dict = {
	0: 'none',
	1: 'max_pool_3x3',
	2: 'avg_pool_3x3',
	3: 'skip_connect',
	4: 'sep_conv_3x3',
	5: 'sep_conv_5x5',
	6: 'dil_conv_3x3',
	7: 'dil_conv_5x5'
	}
    darts_arch = [[], []]
    i=0
    for cell in arch:
        for n in cell:
            darts_arch[i].append((op_dict[n[1]], n[0]))
        i += 1
    print('Genotype(normal=%s, normal_concat=[2,3,4,5], reduce=%s, reduce_concat=[2,3,4,5])' % (str(darts_arch[0]), str(darts_arch[1])))

if __name__=="__main__":
    args = sys.argv[1:]
    print(args[0])
    arch = ast.literal_eval(args[0])
    parse_arch_to_darts('cnn', arch)
