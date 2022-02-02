import json
import powerlaw
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


language = 'es'

data_wexea = json.load(open('/media/michi/Data/wikipedia/statistic_data/data_wexea_' + language + '.json'))
data_wiki = json.load(open('/media/michi/Data/wikipedia/statistic_data/data_wiki_' + language + '.json'))


wexea = []
wiki = []

relation_idx = {}
idx = 1

for relation in data_wexea:
    if relation not in relation_idx:
        relation_idx[relation] = idx
        idx += 1
    wexea.append((relation,data_wexea[relation]))

for relation in data_wiki:
    wiki.append((relation, data_wiki[relation]))

wexea.sort(key=lambda x:x[1],reverse=True)
wiki.sort(key=lambda x:x[1],reverse=True)

relation_wiki = {}
relation_wexea = {}

y_wiki = []
y_wexea = []
x = []
width = []
idx = 1
rs = set()
for tuple in wexea:
    relation = tuple[0]

    relation_name = relation.split('/')[-1][:-1]

    if relation in data_wiki:
        c_wiki = data_wiki[relation]
        c_wexea = data_wexea[relation]
        r_idx = relation_idx[relation]

        if c_wiki >= 100 and c_wexea >= 100:
            rs.add(relation)
            x.append(idx)
            idx += 1
            y_wiki.append(c_wiki)
            y_wexea.append(c_wexea)
            width.append(relation_name)

print(len(rs))

#plt.bar(x,np.array(y_wexea) - np.array(y_wiki))
plt.rcParams["figure.figsize"] = (15,8)
b = 50
if b != -1:
    plt.bar(x[:b],y_wexea[:b],)
    plt.bar(x[:b],y_wiki[:b])
    plt.xticks(x[:b], labels=width[:b], rotation=270)
    plt.subplots_adjust(bottom=0.26)
    plt.ylabel("Number of sentences", fontsize=18)
else:
    plt.bar(x, y_wexea, )
    plt.bar(x, y_wiki)
    plt.yscale('log')
    plt.ylabel("Number of sentences (log scale)", fontsize=18)
#plt.ylim([0, 800000])
plt.xlabel("Distinct relations (sorted)", fontsize=18)

plt.show()


'''relations_list = []
relation_c = {}
idx = 0
for tuple in wexea:
    relation = tuple[0]
    if relation in data_wiki:
        if relation not in relation_c:
            relation_c[relation] = idx
            idx += 1
        relations_list.append(relation_c[relation])'''


'''for tuple in wexea:
    relation = tuple[0]
    if relation in data_wiki:
        print(relation + '\t' + str(data_wiki[relation]) + '\t' + str(data_wexea[relation]))

'''

'''fit = powerlaw.Fit(np.array(relations_list)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')

print('alpha= ',fit.power_law.alpha,'  sigma= ',fit.power_law.sigma)'''


'''y = np.nonzero(relations_list)
x = np.nonzero(y)[0]
plt.bar(x,y)
plt.show()'''