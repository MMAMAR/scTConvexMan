import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE, KLD
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from spektral.layers import TAGConv
from tensorflow.keras.initializers import GlorotUniform
from layers import *
import tensorflow_probability as tfp
import tensorflow as tf
import numpy.random as nprd
from tqdm import tqdm
########################################################################

from scipy.stats import mode
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

########################################################################
import os
import csv
import math
import random
from sklearn import metrics
from graph_function import *
from loss import ZINB, dist_loss
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from intrinsic_dimension import estimate, block_analysis
from sklearn.cluster import SpectralClustering

def broadcast_tensor(A, B):
    if A.shape[0] >= B.shape[0]:
        A = tf.tile(A, tf.constant([3, 1], tf.int32))
        B = tf.tile(B, tf.constant([int(A.shape[0] / B.shape[0]), 1], tf.int32))
        if A.shape[0] % B.shape[0] != 0:
            difference_length = np.arange(A.shape[0] - B.shape[0]).tolist()
            B = tf.concat([B, tf.gather(B, indices=difference_length, axis=0)], axis=0)
    else:
        B = tf.tile(B, tf.constant([3, 1], tf.int32))
        A = tf.tile(A, tf.constant([int(B.shape[0] / A.shape[0]), 1], tf.int32))
        if B.shape[0] % A.shape[0] != 0:
            difference_length = np.arange(B.shape[0] - A.shape[0]).tolist()
            A = tf.concat([A, tf.gather(A, indices=difference_length, axis=0)], axis=0)
    return A, B

#Compute clustering accuracy
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

def thresholding(p, beta_1=0, beta_2=0.9):
    '''
    Thresholding filter to identify datapoints with high-confidence clustering assignment
    beta_2: the threshold over which the soft assignment is considered  reliable(epsilon in Algorithm 1 of the paper)
    '''
    unconf_indices = []
    conf_indices = []
    p = p.numpy()
    confidence1 = p.max(1)
    confidence2 = np.zeros((p.shape[0],))
    a = np.argsort(p, axis=1)[:, -2]
    for i in range(p.shape[0]):
        confidence2[i] = p[i, a[i]]
        if (confidence1[i] > beta_1) and (confidence1[i] - confidence2[i]) > beta_2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices


def generate_unconflicted_data_index_oc(q, beta_2=0.9):
    unconf, conf = thresholding(q, beta_2=beta_2)
    return unconf, conf

def sum_to_x(n, x):
    values = [0.0, x] + list(np.random.uniform(low=0.0, high=x, size=n-1))
    values.sort()
    return [values[i+1] - values[i] for i in range(n)]
   
def interpolate_samples(Z, unconflicted_ind, conflicted_ind, q):
    pred_labels = q.numpy().argmax(1)
    Z_unconf = tf.gather(Z, indices=unconflicted_ind.tolist(), axis=0)
    Z_conf = tf.gather(Z, indices=conflicted_ind.tolist(), axis=0)
    pred_labels_unconf = pred_labels[unconflicted_ind.tolist()]
    pred_labels_conf = pred_labels[conflicted_ind.tolist()]
    k  = q.shape[1]
    dictionary_unconf = {}
    dictionary_conf = {}
    for i in range(k):
        dictionary_unconf[i] = []
        dictionary_conf[i] = []
    for i in range(len(Z_unconf)):
        dictionary_unconf[pred_labels_unconf[i]] += [i]
    for i in range(len(Z_conf)):
        dictionary_conf[pred_labels_conf[i]] += [i]

    z_dict_uncof = {}
    empty_clusters = set()
    for j in range(k):
        if len(dictionary_unconf[j]) > 0:
            z_dict_uncof[j] = tf.gather(Z_unconf, indices=dictionary_unconf[j], axis=0)
        else:
        	empty_clusters.add(j)

    z_dict_cof = {}
    for j in range(k):
        if len(dictionary_conf[j]) > 0:
            z_dict_cof[j] = tf.gather(Z_conf, indices=dictionary_conf[j], axis=0)
        else:
        	empty_clusters.add(j)

    for j in empty_clusters:
    	if j in z_dict_uncof.keys():
    		del z_dict_uncof[j]
    	if j in z_dict_cof.keys():
    		del z_dict_cof[j]
    
    for j in z_dict_uncof.keys():
        z_dict_uncof[j], z_dict_cof[j] = broadcast_tensor(z_dict_uncof[j], z_dict_cof[j])

    Z_unconf_interpolated = {}
    for j in z_dict_uncof.keys():
        z0 = z_dict_uncof[j]
        indx = np.arange(len(z0))
        np.random.shuffle(indx)
        z1 = tf.gather(z0, indices=indx.tolist(), axis=0)
        np.random.shuffle(indx)
        z2 = tf.gather(z0, indices=indx.tolist(), axis=0)
        conv_combinations = [sum_to_x(3, 1) for i in range(len(z0))]
        conv_combinations = np.array(conv_combinations)
        alpha_0, alpha_1, alpha_2 = conv_combinations[:, 0].reshape(-1, 1), conv_combinations[:, 1].reshape(-1, 1), conv_combinations[:, 2].reshape(-1, 1)
        z_interp_unconf = alpha_0 * z0 + alpha_1 * z1 + alpha_2 * z2
        Z_unconf_interpolated[j] = z_interp_unconf


    Z_unconf_conf_interpolated = {}
    for j in z_dict_uncof.keys():
        z0_unconf = z_dict_cof[j]
        indx_conf = np.arange(len(z0_unconf))
        np.random.shuffle(indx_conf)
        z1_unconf = tf.gather(z0_unconf, indices=indx_conf.tolist(), axis=0)
        z2_conf_ = z_dict_uncof[j]
        indx_unconf = np.arange(len(z2_conf_))
        z2_conf_new = tf.gather(z2_conf_, indices=indx_unconf.tolist(), axis=0)
        conv_combinations = [sum_to_x(3, 1) for i in range(len(z0_unconf))]
        conv_combinations = np.array(conv_combinations)
        alpha_0, alpha_1, alpha_2 = conv_combinations[:, 0].reshape(-1, 1), conv_combinations[:, 1].reshape(-1,1), conv_combinations[:,2].reshape(-1, 1)
        z_interp_conf_unconf = alpha_0 * z0_unconf + alpha_1 * z1_unconf + alpha_2 * z2_conf_new
        Z_unconf_conf_interpolated[j] = z_interp_conf_unconf
    return Z_unconf_interpolated, Z_unconf_conf_interpolated, z_dict_uncof, z_dict_cof 

def slice_x(x, x_hat, unconflicted_ind, q):
    pred_labels = q.numpy().argmax(1)
    x_unconf = x[unconflicted_ind.tolist()]
    x_hat_unconf = tf.gather(x_hat, indices=unconflicted_ind.tolist(), axis=0)
    pred_labels_unconf = pred_labels[unconflicted_ind.tolist()]
    k = q.shape[1]
    dict = {}
    for i in range(k):
        dict[i] = []
    for i in range(len(x_unconf)):
        dict[pred_labels_unconf[i]] += [i]
    x_dict_uncof = {}
    for j in range(k):
        if len(dict[j]) > 0:
            x_dict_uncof[j] = x_unconf[dict[j]]
    x_hat_dict_uncof = {}
    for j in range(k):
        if len(dict[j]) > 0:
            x_hat_dict_uncof[j] = tf.gather(x_hat_unconf, indices=dict[j], axis=0)
    X_true_clusters = {}
    for j in x_dict_uncof.keys():
        gamma = random.uniform(0, 1)
        gamma = 0.5 - np.abs(gamma - 0.5)
        x_true_j = gamma * x_dict_uncof[j] + (1 - gamma) * x_hat_dict_uncof[j]
        X_true_clusters[j] = x_true_j
    return X_true_clusters

def create_discriminator(out_dim, discr_dims, name,act):
    discr_in = Input(shape=out_dim)
    h = Dense(units=discr_dims[0], activation=act)(discr_in)
    discr_out = Dense(units=1, activation="sigmoid")(h)
    discriminator = Model(inputs=discr_in, outputs=discr_out, name=name)
    return discriminator

def computeID(emb_np, nres=10, fraction=1):
    ID = []
    n = int(np.round(emb_np.shape[0] * fraction))            
    dist = squareform(pdist(emb_np, 'euclidean'))
    for i in range(nres):
        dist_s = dist
        perm = np.random.permutation(emb_np.shape[0])[0:n]
        dist_s = dist_s[perm,:]
        dist_s = dist_s[:,perm]
        ID.append(estimate(dist_s)[2]) 
    return ID

def compute_LID(emb_np, th=0.99):
    scaler = StandardScaler()
    scaler.fit(emb_np)
    embn = scaler.transform(emb_np)
    pca = PCA()
    pca.fit(embn)
    sv = pca.singular_values_
    evr = pca.explained_variance_ratio_
    cs = np.cumsum(pca.explained_variance_ratio_)
    return np.argwhere(cs > th)[0][0]

class SCTCM(tf.keras.Model):

    def __init__(self, X,  adj, adj_n, y, hidden_dim=128, latent_dim=15, dec_dim=None, adj_dim=32, act="relu"):
        super(SCTCM, self).__init__()
        if dec_dim is None:
            dec_dim = [128, 256, 512]
            #dec_dim = [128, 256]
        self.latent_dim = latent_dim
        self.X = X
        self.adj = np.float32(adj)
        self.adj_n = np.float32(adj_n)
        self.y = y
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.sparse = False

        initializer = GlorotUniform(seed=7)
        '''
        The architecture of the graph autoencoder. This piece of code is similar to scTAG(https://github.com/Philyzh8/scTAG)  
        '''
        # The graph convolutionl encoder
        X_input = Input(shape=self.in_dim)
        h = Dropout(0.2)(X_input)
        self.sparse = True
        A_in = Input(shape=self.n_sample, sparse=True)
        h = TAGConv(channels=hidden_dim, kernel_initializer=initializer, activation=act)([h, A_in])
        z_mean = TAGConv(channels=latent_dim, kernel_initializer=initializer)([h, A_in])
        self.encoder = Model(inputs=[X_input, A_in], outputs=z_mean, name="encoder")
        clustering_layer = ClusteringLayer(name='clustering')(z_mean)
        overclustering_layer = ClusteringLayer(name='overclustering')(z_mean)
        self.cluster_model = Model(inputs=[X_input, A_in], outputs=clustering_layer, name="cluster_encoder")
        self.overcluster_model = Model(inputs=[X_input, A_in], outputs=overclustering_layer, name="overcluster_encoder")

        # Adjacency matrix decoder
        dec_in = Input(shape=latent_dim)
        h = Dense(units=adj_dim, activation=None, name="decoder_layer_0")(dec_in)
        h = Bilinear()(h)
        dec_out = Lambda(lambda z: tf.nn.sigmoid(z))(h)
        self.decoderA = Model(inputs=dec_in, outputs=dec_out, name="decoderA")

        #The gene count matrix decoder based on the Zero-Inflated Negative Binomial Model (ZINB)
        decx_in = Input(shape=latent_dim)
        h = Dense(units=dec_dim[0], activation=act, name="decoder_layer_1")(decx_in)
        h = Dense(units=dec_dim[1], activation=act, name="decoder_layer_2")(h)
        h = Dense(units=dec_dim[2], activation=act, name="decoder_layer_3")(h)

        pi = Dense(units=self.in_dim, activation='sigmoid', kernel_initializer='glorot_uniform', name='pi')(h)

        disp = Dense(units=self.in_dim, activation=DispAct, kernel_initializer='glorot_uniform', name='dispersion')(h)

        mean = Dense(units=self.in_dim, activation=MeanAct, kernel_initializer='glorot_uniform', name='mean')(h)

        self.decoderX = Model(inputs=decx_in, outputs=[pi, disp, mean], name="decoderX")
        self.disc_list = [create_discriminator(self.X.shape[1], [64], name = 'discriminator_' + str(cl), act=act) for cl in range(len(set(self.y)))]

    def pre_train(self, epochs=1000, info_step=10, lr=1e-4, W_a=0.3, W_x=1, W_d=0, min_dist=0.5, max_dist=20, save_path='', save=True):
        '''
        The pretraining step optimizes TWO loss functions: A graph reconstruction loss and a ZINB loss
        '''
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if self.sparse == True:
            self.adj_n = tfp.math.dense_to_sparse(self.adj_n)

        if save == True:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            logfile = open(save_path + '/log_pretrain.csv', 'w')
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'ID_global_mean', 'ID_local_mean', 'LID_mean', 'ID_global_error', 'ID_local_error'])
            logwriter.writeheader()

        # Training
        for epoch in range(1, epochs + 1):
            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder([self.X, self.adj_n])
                # X_out = self.decoderX(z)
                pi, disp, mean = self.decoderX(z)
                A_out = self.decoderA(z)

                if W_d:
                    Dist_loss = tf.reduce_mean(dist_loss(z, min_dist, max_dist=max_dist))
                A_rec_loss = tf.reduce_mean(MSE(self.adj, A_out))
                zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.X, mean, mean=True)
                loss = W_a * A_rec_loss + W_x * zinb_loss
                if W_d:
                    loss += W_d * Dist_loss

            vars = self.trainable_weights
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))
            if epoch % info_step == 0:
                if W_d:
                    print("Epoch", epoch, " zinb_loss:", zinb_loss.numpy(), "  A_rec_loss:", A_rec_loss.numpy(),
                         "Dist_loss:", Dist_loss.numpy())
                else:
                    print("Epoch", epoch, " zinb_loss:", zinb_loss.numpy(), "  A_rec_loss:", A_rec_loss.numpy())

        print("Pre_train Finish!")

    def alt_train(self, y, epochs=3000, lr_dis=0.0001, lr_gen=0.0001, lr_clus=5e-4, lr_overclus=5e-4, W_a=3, W_x=1, W_c=1.5, W_oc =0.5, info_step=8, n_update=8, centers=None,  overcenters=None, save_path='./saving_path', save = False, threshold_2=0.7, threshold_3=0.3, beta_1=0.95, beta_2=0.95, beta_3=0.95, beta_4=0.95, old=False, gen_freq=100, cluster_freq=2, overcluster_freq=2):
        '''
        The clustering step optimizes FIVE loss functions: A graph reconstruction loss, a ZINB-based loss, KL-divergence loss, a generator loss and discriminator loss
        '''
        self.cluster_model.get_layer(name='clustering').clusters = centers
        self.overcluster_model.get_layer(name='overclustering').clusters = overcenters

        # Training
        optimizer_1 = tf.keras.optimizers.Adam(learning_rate=lr_dis)
        optimizer_2 = tf.keras.optimizers.Adam(learning_rate=lr_gen)
        optimizer_3 = tf.keras.optimizers.Adam(learning_rate=lr_clus, beta_1=beta_3)
        optimizer_4 = tf.keras.optimizers.Adam(learning_rate=lr_overclus, beta_1=beta_4)
        if save == True:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            logfile = open(save_path + '/log_train.csv', 'w')

            logwriter = csv.DictWriter(logfile, fieldnames=['iter','acc', 'nmi', 'ari', 'unconf_acc', 'conf_acc',
                                                   'nb_unconf', 'nb_conf', 'nb_added_links',
                                                   'nb_false_added_links',
                                                   'nb_true_added_links', 'nb_dropped_links', 'nb_false_dropped_links',
                                                   'nb_true_dropped_links','nb_unconf','nb_conf', 'ID_global_mean', 'ID_local_mean', 'LID_mean', 'ID_global_error', 'ID_local_error'])

            logwriter.writeheader()
            count_target_links = {"nb_added_links": 0,
                                  "nb_false_added_links": 0,
                                  "nb_true_added_links": 0,
                                  "nb_deleted_links": 0,
                                  "nb_false_deleted_links": 0,
                                  "nb_true_deleted_links": 0}

        adj_norm_pos = self.adj_n
        adj_pos = self.adj
        print('Training in progress ...............')
        for epoch in range(0, epochs):
            with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape, tf.GradientTape() as c_tape, tf.GradientTape() as oc_tape:
                z = self.encoder([self.X, adj_norm_pos])
                q_out = self.cluster_model([self.X, adj_norm_pos])
                q_out_overlclustering = self.overcluster_model([self.X, adj_norm_pos])
                pi, disp, mean = self.decoderX(z)
                A_out = self.decoderA(z)

                if epoch % n_update == 0:
                    labels_matrix = np.zeros_like(q_out.numpy())
                    labels_matrix[np.arange(len(q_out)), q_out.numpy().argmax(1)] = 1.0
                    labels_matrix_oc = np.zeros_like(q_out_overlclustering.numpy())
                    labels_matrix_oc[np.arange(len(q_out_overlclustering)), q_out_overlclustering.numpy().argmax(1)] = 1.0

                    '''Compute unconflicted and conflicted lists'''
                    unconflicted_ind, conflicted_ind = generate_unconflicted_data_index_oc(q_out, beta_2=threshold_2)

                    if epoch <= 40:
                        over_unconflicted_ind, over_conflicted_ind = generate_unconflicted_data_index_oc(q_out_overlclustering, beta_2=threshold_3)
                        p_overclustering = self.target_distribution(q_out_overlclustering, over_unconflicted_ind, over_conflicted_ind)

                    p = self.target_distribution(q_out, unconflicted_ind, conflicted_ind)
                    if old == False:
                        '''Update the input graph each n_update steps'''
                        adj_pos, adj_norm_pos = self.update_graph(unconflicted_ind)
                        adj_pos = adj_pos.toarray()
                        adj_norm_pos = tf.sparse.from_dense(adj_norm_pos.toarray())


                # Compute a convex combination
                z_unconf_interpolated, z_unconf_conf_interpolated, z_dict_uncof, z_dict_cof = interpolate_samples(z, unconflicted_ind, conflicted_ind, q_out)
                interp_decoded_unconf = {}
                for j in z_unconf_interpolated.keys():
                    _, _, interp_decoded_unconf[j] = self.decoderX(z_unconf_interpolated[j])

                interp_decoded_unconf_conf = {}
                for j in z_unconf_conf_interpolated.keys():
                    _, _, interp_decoded_unconf_conf[j] = self.decoderX(z_unconf_conf_interpolated[j])

                decoded_unconf = {}
                for j in z_dict_uncof.keys():
                    _, _, decoded_unconf[j] = self.decoderX(z_dict_uncof[j])

                decoded_conf = {}
                for j in z_dict_cof.keys():
                    _, _, decoded_conf[j] = self.decoderX(z_dict_cof[j])

                discr_losses = []
                gen_losses = []
                for j in interp_decoded_unconf.keys():
                    disc_dataset_j = tf.concat([decoded_unconf[j], decoded_conf[j]], axis=0)
                    true_dis_labels_j = tf.constant(np.array(len(decoded_unconf[j]) * [1] + len(decoded_conf[j]) * [0]).reshape(-1, 1), dtype=tf.float32)
                    loss_discr_j = tf.math.reduce_mean(tf.keras.backend.binary_crossentropy(true_dis_labels_j, self.disc_list[j](disc_dataset_j)))
                    discr_losses.append(loss_discr_j)
                    true_gen_labels_j = tf.constant(np.array(len(interp_decoded_unconf[j]) * [1]).reshape(-1, 1), dtype=tf.float32)
                    gen_losses.append(tf.math.reduce_mean(tf.keras.backend.binary_crossentropy(true_gen_labels_j, self.disc_list[j](interp_decoded_unconf[j]))))


                discr_loss = sum(discr_losses)
                gen_loss = sum(gen_losses)

                A_rec_loss = tf.reduce_mean(MSE(adj_pos, A_out))
                zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.X, mean, mean=True)
                cluster_loss = tf.reduce_mean(KLD(q_out, p))
                overcluster_loss = tf.reduce_mean(KLD(q_out_overlclustering, p_overclustering))
                clustering_loss = W_a * A_rec_loss + W_x * zinb_loss + W_c * cluster_loss 
                overclustering_loss = W_a * A_rec_loss + W_x * zinb_loss + W_oc * overcluster_loss

            vars_disc = self.disc_list.trainable_weights
            vars_gen = [w for w in self.trainable_weights if w.name not in [wt.name for wt in self.disc_list.trainable_weights]]
            # The discriminator loss optimization
            grads_disc = d_tape.gradient(discr_loss, vars_disc)
            optimizer_1.apply_gradients(zip(grads_disc, vars_disc))

            # The clustering loss optimization
            if epoch % cluster_freq == 0:
                grads_c = c_tape.gradient(clustering_loss, vars_gen)
                optimizer_3.apply_gradients(zip(grads_c, vars_gen))
            # The overclustering loss optimization
            if epoch % overcluster_freq == 1:
               grads_oc = oc_tape.gradient(overclustering_loss, vars_gen)
               optimizer_4.apply_gradients(zip(grads_oc, vars_gen))
            # The generator loss optimization
            if epoch % gen_freq == gen_freq-1  or epoch % gen_freq == gen_freq-2 or epoch % gen_freq == gen_freq-3 or epoch % gen_freq == gen_freq-4 or epoch % gen_freq == gen_freq-5:
            	grads_gen = g_tape.gradient(gen_loss, vars_gen)
            	optimizer_2.apply_gradients(zip(grads_gen, vars_gen))


            if epoch % info_step == 0:
                pred = tf.math.argmax(q_out,1).numpy()
                acc = np.round(cluster_acc(y, pred), 5) # ind
                acc_unconflicted = np.round(cluster_acc(y[unconflicted_ind], pred[unconflicted_ind]), 5) # ind
                acc_conflicted = np.round(cluster_acc(y[conflicted_ind], pred[conflicted_ind]), 5) #ind
                y = np.array(list(map(int, y)))
                nmi = np.round(metrics.normalized_mutual_info_score(y, pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, pred), 5)

                ID = []
                PC_ID = []
                z_np = z.numpy().astype(np.float64)
                for k in range(len(set(self.y))):
                	if sum(pred==k) > 20:
                		ID.append(computeID(z_np[pred==k]))
                		PC_ID.append(compute_LID(z_np[pred==k]))
                ID = np.asarray(ID)
                PC_ID = np.asarray(PC_ID)
                ID_global_mean = np.mean(ID)
                ID_local_mean = np.mean(ID, axis=1)
                ID_global_error = np.std(ID)
                ID_local_error = np.std(ID, axis=1)
                PC_ID_mean = np.mean(PC_ID)

                if save == True:
                    logdict = dict(iter=epoch,
                                   acc=acc, nmi=nmi, ari=ari,
                                   unconf_acc=acc_unconflicted, conf_acc=acc_conflicted,
                                   nb_added_links=count_target_links["nb_added_links"],
                                   nb_false_added_links=count_target_links["nb_false_added_links"],
                                   nb_true_added_links=count_target_links["nb_true_added_links"],
                                   nb_dropped_links=count_target_links["nb_deleted_links"],
                                   nb_false_dropped_links=count_target_links["nb_false_deleted_links"],
                                   nb_true_dropped_links=count_target_links["nb_true_deleted_links"],
                                   nb_unconf=len(unconflicted_ind), nb_conf=len(conflicted_ind),
                                   ID_global_mean=ID_global_mean, ID_local_mean=ID_local_mean, 
                                   LID_mean=PC_ID_mean, ID_global_error=ID_global_error, ID_local_error=ID_local_error)
                    logwriter.writerow(logdict)
                    logfile.flush()

        print('Training completed!')
        tf.compat.v1.disable_eager_execution()
        q = tf.constant(q_out)
        session = tf.compat.v1.Session()
        q = session.run(q)
        self.y_pred = q.argmax(1)
        self.emb = z
        return self, acc, nmi, ari

    def target_distribution(self, p,unconflicted_ind, conflicted_ind):
        '''
        q[i] = p[i] if i belongs to the conflicted indexes
        q[i,argmax(p[i])] = 1 and q[i,j≠argmax(p[i])] = 0 if i belongs to the belongs to the unconflicted indexes
        '''
        p = p.numpy()
        q = np.zeros(p.shape)
        q[conflicted_ind] = p[conflicted_ind]
        q[unconflicted_ind, np.argmax(p[unconflicted_ind], axis=1)] = 1
        q = tf.convert_to_tensor(q,dtype=tf.float32)
        return q

    def target_distribution_square(self, p):
        weight = tf.math.divide(tf.math.square(p), tf.math.reduce_sum(p, axis=0, keepdims=True))
        return tf.transpose(tf.math.divide(tf.transpose(weight), tf.math.reduce_sum(weight, axis=1)))

    def embedding(self, count, adj_n):
        if self.sparse and not isinstance(adj_n,tf.sparse.SparseTensor):
            adj_n = tfp.math.dense_to_sparse(adj_n)
        return np.array(self.encoder([count, adj_n]))

    def rec_A(self, count, adj_n):
        h = self.encoder([count, adj_n])
        rec_A = self.decoderA(h)
        return np.array(rec_A)

    def get_label(self, count, adj_n):
        if self.sparse and not isinstance(adj_n,tf.sparse.SparseTensor):
            adj_n = tfp.math.dense_to_sparse(adj_n)
        clusters = self.cluster_model([count, adj_n]).numpy()
        labels = np.array(clusters.argmax(1))
        return labels.reshape(-1, )

    def update_graph(self, unconf_indices):
        '''
        Update the input graph in order to make it clustering relevant
        '''
        y_pred = self.get_label(self.X,self.adj_n)
        adj_pos = sp.lil_matrix(self.adj)
        idx = unconf_indices[self.generate_centers(unconf_indices)]
        for i, k in enumerate(unconf_indices):
         adj_k_pos = adj_pos[k].tocsr().indices
         if not(np.isin(idx[i], adj_k_pos)) and (y_pred[k] == y_pred[idx[i]]):
             adj_pos[k, idx[i]] = 1
        adj_pos = adj_pos - sp.dia_matrix((adj_pos.diagonal()[np.newaxis, :], [0]), shape=adj_pos.shape)
        adj_pos = adj_pos.tocsr()
        adj_pos.eliminate_zeros()
        adj_norm_pos = norm_adj(adj_pos)
        return adj_pos, adj_norm_pos

    def generate_centers(self, unconf_indices):
        y_pred = self.get_label(self.X, self.adj_n)[unconf_indices]
        emb_unconf = self.embedding(self.X, self.adj_n)[unconf_indices]
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(emb_unconf)
        _, indices = nn.kneighbors(self.cluster_model.get_layer(name='clustering').clusters)
        return indices[y_pred]
