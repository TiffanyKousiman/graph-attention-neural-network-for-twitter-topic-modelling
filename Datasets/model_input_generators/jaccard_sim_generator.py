import numpy as np
input_path = '../model_input_data/'

for tag in ['2013_2017', '2018_2022']:
    # read cleaned tweets
    docs = []
    print('calculate doc-sim matrix')
    with open(input_path + f'cleaned_tweets_{tag}.csv', 'r',encoding="ISO-8859-1") as f:
        for itm in f:
            docs.append(itm)
    docs = [d.split('\n')[0] for d in docs] #remove \n separator
    n_docs = len(docs)

    # compute jaccard similarity
    def get_jaccard_sim(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        if (len(a) + len(b) - len(c)) == 0:
            out = 0
        else:
            out = float(len(c)) / (len(a) + len(b) - len(c))
        return out

    doc_dis = np.zeros([n_docs,n_docs])
    y = len(docs)
    x = 0
    for i in range(0,len(docs)):
        if y>0:
            for j in range(x,y):
                sim = get_jaccard_sim(docs[i],docs[j])
                doc_dis[i][j] = sim
                doc_dis[j][i] = sim
        else:
            break
        x+=1

    # save jac distance matrix
    np.savetxt(input_path + f"doc_doc_t_jaccard_{tag}.txt", doc_dis, fmt='%f')