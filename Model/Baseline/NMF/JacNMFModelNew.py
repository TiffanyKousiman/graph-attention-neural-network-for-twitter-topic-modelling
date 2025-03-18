import time
import numpy as np
from numpy.linalg import norm
import random
import csv
class JacNMFModel(object):
    def __init__(
            self,
            A, S,
            IH1=[], IH2=[], IW=[],
            alpha=1.0,
            #beta=0.1,
            n_topic=10, max_iter=500, max_err=1e-2,
            rand_init=True, fix_seed=False):

        if fix_seed:
            np.random.seed(0)

        self.A = A
        self.S = S

        self.n_row = A.shape[0]
        self.n_col = A.shape[1]

        self.n_topic = n_topic
        self.max_iter = max_iter
        self.alpha = alpha
        #self.beta = beta
        #self.B = np.ones([self.n_topic, 1])
        self.B = np.ones([self.n_topic, 1])
        self.max_err = max_err

        if rand_init:
            self.nmf_init_rand()
        else:
            self.nmf_init(IH1, IH2, IW)
        #with open()
        self.nmf_iter()

    def nmf_init_rand(self):
        self.H1 = np.random.random((self.n_col, self.n_topic))
        self.H2 = np.random.random((self.n_col, self.n_topic))
        self.W = np.random.random((self.n_row, self.n_topic))

        for k in range(self.n_topic):
            self.H1[:, k] /= norm(self.H1[:, k])
            self.H2[:, k] /= norm(self.H2[:, k])

    def nmf_init(self, IH1, IH2, IW):
        self.H1 = IH1
        self.H2 = IH2
        self.W = IW

        for k in range(self.n_topic):
            self.H1[:, k] /= norm(self.H1[:, k])
            self.H2[:, k] /= norm(self.H2[:, k])

    def nmf_iter(self):
        loss_old = 1e20
        print('loop begin')
        start_time = time.time()
        # n = random.randint(0, 1000)
        # file_name = "Iter_info_" + str(n)
        # with open(file_name, 'w') as f:
        print('loop begin')
        start_time = time.time()
        for i in range(100):
            self.nmf_solver()
            loss = self.nmf_loss()
            if loss_old - loss < self.max_err:
                break
            loss_old = loss
            end_time = time.time()
            print('Step={}, Loss={}, Time={}s'.format(i, loss, end_time - start_time))
            # f.write(str(loss))
        # for i in range(100):
        #     self.nmf_solver()
        #     loss = self.nmf_loss()
        #     if loss_old - loss < self.max_err:
        #         break
        #     loss_old = loss
        #     end_time = time.time()
        #     print('Step={}, Loss={}, Time={}s'.format(i, loss, end_time - start_time))
            # row = "SeaNMF,"+dataset+","+str(len(predicted))+","+str(args.n_topics)+","+str(args.alpha)+","+str(args.max_err)+","+str(time_taken)+","+str(df1)+","+str(dnmi)


    def nmf_solver(self):
        epss = 1e-40
        H1tH1 = np.dot(self.H1.T, self.H1)
        AH1 = np.dot(self.A, self.H1)
        for k in range(self.n_topic):
            self.W[:, k] = self.W[:, k] + AH1[:, k] - np.dot(self.W, H1tH1[:, k])
            self.W[:, k] = np.maximum(self.W[:, k], epss)

        AtW = np.dot(self.A.T, self.W)
        SH1 = np.dot(self.S, self.H1)
        #SH1 = np.dot(self.S, self.H2)
        H1tH1 = np.dot(self.H1.T, self.H1)
        #H1tH1 = np.dot(self.H2.T, self.H2)
        WtW = np.dot(self.W.T, self.W)
        W1 = self.H1.dot(self.B)
        #W1 = self.H2.dot(self.B)

        for k in range(self.n_topic):
            num0 = WtW[k, k] * self.H1[:, k] + self.alpha * (H1tH1[k, k] * self.H1[:, k])
            num1 = AtW[:, k] + self.alpha * SH1[:, k]
            num2 = np.dot(self.H1, WtW[:, k]) + self.alpha * np.dot(self.H1, H1tH1[:, k])
                   #+ self.beta * W1[0]
            self.H1[:, k] = num0 + num1 - num2
            self.H1[:, k] = np.maximum(self.H1[:, k], epss)  # project > 0
            self.H1[:, k] /= norm(self.H1[:, k]) + epss  # normalize

        H1tH1 = self.H1.T.dot(self.H1)
        StH1 = np.dot(self.S, self.H1)
        for k in range(self.n_topic):
            self.H2[:, k] = self.H2[:, k] + StH1[:, k] - np.dot(self.H2, H1tH1[:, k])
            self.H2[:, k] = np.maximum(self.H2[:, k], epss)

    def nmf_loss(self):
        '''
        Calculate loss
        '''
        loss = norm(self.A - np.dot(self.W, np.transpose(self.H1)), 'fro') ** 2 / 2.0
        if self.alpha > 0:
            loss += self.alpha * norm(np.dot(self.H1, np.transpose(self.H2)) - self.S, 'fro') ** 2 / 2.0
        #if self.beta > 0:
         #   loss += self.beta * norm(self.H1, 1) ** 2 / 2.0

        #loss += self.beta * norm(self.W, 1) ** 2 / 2.0  # L1 regularisation
        return loss

    def get_lowrank_matrix(self):
        return self.H1, self.H2, self.W

    def save_format(self, H1file='H1.txt', H2file='H2.txt', Wfile='W.txt'):
        np.savetxt(H1file, self.H1)
        np.savetxt(H2file, self.H2)
        np.savetxt(Wfile, self.W)

