from gae_layers import GraphConvolution, InnerProductDecoder
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class GCNModelAE(Model):
    def __init__(self, num_drug, num_protein, num_disease, num_sideeffect,
                 dim_drug, dim_protein, dim_disease, dim_sideeffect,
                 hidden_dim = 100, out_dim = 100, dropout = 0.1, 
                 version = 'gae', **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.num_drug = num_drug
        self.num_protein = num_protein
        self.num_disease = num_disease
        self.num_sideeffect = num_sideeffect

        self.dim_drug = dim_drug
        self.dim_protein = dim_protein
        self.dim_disease = dim_disease
        self.dim_sideeffect = dim_sideeffect

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.version = version

        ### data holder
        self.drug_drug = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_drug_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.drug_chemical = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_chemical_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.drug_disease = tf.placeholder(tf.float32, [num_drug, num_disease])
        self.drug_disease_normalize = tf.placeholder(tf.float32, [num_drug, num_disease])

        self.drug_sideeffect = tf.placeholder(tf.float32, [num_drug, num_sideeffect])
        self.drug_sideeffect_normalize = tf.placeholder(tf.float32, [num_drug, num_sideeffect])

        self.protein_protein = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_protein_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])

        self.protein_sequence = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_sequence_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])

        self.protein_disease = tf.placeholder(tf.float32, [num_protein, num_disease])
        self.protein_disease_normalize = tf.placeholder(tf.float32, [num_protein, num_disease])

        self.disease_drug = tf.placeholder(tf.float32, [num_disease, num_drug])
        self.disease_drug_normalize = tf.placeholder(tf.float32, [num_disease, num_drug])

        self.disease_protein = tf.placeholder(tf.float32, [num_disease, num_protein])
        self.disease_protein_normalize = tf.placeholder(tf.float32, [num_disease, num_protein])

        self.sideeffect_drug = tf.placeholder(tf.float32, [num_sideeffect, num_drug])
        self.sideeffect_drug_normalize = tf.placeholder(tf.float32, [num_sideeffect, num_drug])

        self.drug_protein = tf.placeholder(tf.float32, [num_drug, num_protein])
        self.drug_protein_normalize = tf.placeholder(tf.float32, [num_drug, num_protein])

        self.protein_drug = tf.placeholder(tf.float32, [num_protein, num_drug])
        self.protein_drug_normalize = tf.placeholder(tf.float32, [num_protein, num_drug])

        self.drug_protein_mask = tf.placeholder(tf.float32, [num_drug, num_protein])

        ### model
        self.build()

    def _build(self):
        ### features
        self.drug_input = weight_variable([self.num_drug,self.dim_drug])
        self.protein_input = weight_variable([self.num_protein,self.dim_protein])
        self.disease_input = weight_variable([self.num_disease,self.dim_disease])
        self.sideeffect_input = weight_variable([self.num_sideeffect,self.dim_sideeffect])

        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.drug_input))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.protein_input))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.disease_input))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.sideeffect_input))
        
        #feature passing weights (maybe different types of nodes can use different weights)
        W0 = weight_variable([self.out_dim + self.dim_drug, self.out_dim])
        b0 = bias_variable([self.out_dim])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0))

        ### embedding 
        ## drug
        drug_embeddings_self =  self.Embed(self.drug_input, self.drug_drug_normalize, self.dim_drug)
        drug_embeddings_chemical = self.Embed(self.drug_input, self.drug_chemical_normalize, self.dim_drug)
        drug_embeddings_disease = self.Embed(self.disease_input, self.drug_disease_normalize, self.dim_disease)
        drug_embeddings_sideeffect = self.Embed(self.sideeffect_input, self.drug_sideeffect_normalize, self.dim_sideeffect)
        drug_embeddings_protein = self.Embed(self.protein_input, self.drug_protein_normalize, self.dim_protein)
        self.drug_embeddings = tf.concat([drug_embeddings_self + 
                                          drug_embeddings_chemical +
                                          drug_embeddings_disease +
                                          drug_embeddings_sideeffect +
                                          drug_embeddings_protein,
                                          self.drug_input], axis = 1)
        self.drug_embeddings = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.drug_embeddings, W0) + b0), dim=1)
        ## protein
        protein_embeddings_self =  self.Embed(self.protein_input, self.protein_protein_normalize, self.dim_protein)
        protein_embeddings_sequence = self.Embed(self.protein_input, self.protein_sequence_normalize, self.dim_protein)
        protein_embeddings_disease = self.Embed(self.disease_input, self.protein_disease_normalize, self.dim_disease)
        protein_embeddings_drug = self.Embed(self.drug_input, self.protein_drug_normalize, self.dim_drug)
        self.protein_embeddings = tf.concat([protein_embeddings_self +
                                             protein_embeddings_sequence +
                                             protein_embeddings_disease +
                                             protein_embeddings_drug,
                                             self.protein_input], axis = 1)
        self.protein_embeddings = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.protein_embeddings, W0) + b0), dim=1)
        ## disease
        disease_embeddings_drug = self.Embed(self.drug_input, self.disease_drug_normalize, self.dim_drug)
        disease_embeddings_protein = self.Embed(self.protein_input, self.disease_protein_normalize, self.dim_protein)
        self.disease_embeddings = tf.concat([disease_embeddings_drug +
                                             disease_embeddings_protein,
                                             self.disease_input], axis = 1)
        self.disease_embeddings = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.disease_embeddings, W0) + b0), dim=1) 
        ## side effect
        sideeffect_embeddings_drug = self.Embed(self.drug_input, self.sideeffect_drug_normalize, self.dim_drug)
        self.sideeffect_embeddings = tf.concat([sideeffect_embeddings_drug,
                                                self.sideeffect_input], axis = 1)
        self.sideeffect_embeddings = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(self.sideeffect_embeddings, W0) + b0), dim=1)

        ### reconstruction
        self.drug_protein_reconstruct = InnerProductDecoder(input_dim=self.out_dim,
                                                            act=lambda x: x,
                                                            logging=self.logging)(self.drug_embeddings, self.protein_embeddings)
        self.drug_drug_reconstruct = InnerProductDecoder(input_dim=self.out_dim,
                                                         act=lambda x: x,
                                                         logging=self.logging)(self.drug_embeddings, None) 
        self.drug_chemical_reconstruct = InnerProductDecoder(input_dim=self.out_dim,
                                                             act=lambda x: x,
                                                             logging=self.logging)(self.drug_embeddings, None)
        self.drug_disease_reconstruct = InnerProductDecoder(input_dim=self.out_dim,
                                                            act=lambda x: x,
                                                            logging=self.logging)(self.drug_embeddings, self.disease_embeddings)
        self.drug_sideeffect_reconstruct = InnerProductDecoder(input_dim=self.out_dim,
                                                               act=lambda x: x,
                                                               logging=self.logging)(self.drug_embeddings, self.sideeffect_embeddings)
        self.protein_protein_reconstruct = InnerProductDecoder(input_dim=self.out_dim,
                                                               act=lambda x: x,
                                                               logging=self.logging)(self.protein_embeddings, None)
        self.protein_sequence_reconstruct = InnerProductDecoder(input_dim=self.out_dim,
                                                                act=lambda x: x,
                                                                logging=self.logging)(self.protein_embeddings, None)
        self.protein_disease_reconstruct = InnerProductDecoder(input_dim=self.out_dim,
                                                               act=lambda x: x,
                                                               logging=self.logging)(self.protein_embeddings, self.disease_embeddings)

        ### loss
        tmp = tf.multiply(self.drug_protein_mask, (self.drug_protein_reconstruct-self.drug_protein))
        self.drug_protein_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp)) 
        self.drug_drug_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_drug_reconstruct-self.drug_drug), 
                                                                    (self.drug_drug_reconstruct-self.drug_drug)))
        self.drug_chemical_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_chemical_reconstruct-self.drug_chemical), 
                                                                        (self.drug_chemical_reconstruct-self.drug_chemical)))
        self.drug_disease_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_disease_reconstruct-self.drug_disease), 
                                                                       (self.drug_disease_reconstruct-self.drug_disease)))
        self.drug_sideeffect_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_sideeffect_reconstruct-self.drug_sideeffect), 
                                                                          (self.drug_sideeffect_reconstruct-self.drug_sideeffect))) 
        self.protein_protein_reconstruct_loss = tf.reduce_sum(tf.multiply((self.protein_protein_reconstruct-self.protein_protein), 
                                                                          (self.protein_protein_reconstruct-self.protein_protein)))
        self.protein_sequence_reconstruct_loss = tf.reduce_sum(tf.multiply((self.protein_sequence_reconstruct-self.protein_sequence), 
                                                                           (self.protein_sequence_reconstruct-self.protein_sequence)))
        self.protein_disease_reconstruct_loss = tf.reduce_sum(tf.multiply((self.protein_disease_reconstruct-self.protein_disease), 
                                                                          (self.protein_disease_reconstruct-self.protein_disease)))
        self.l2_loss = tf.add_n(tf.get_collection("l2_reg")) # l2 regularization loss

        self.loss = self.drug_protein_reconstruct_loss + 1.0*(self.drug_drug_reconstruct_loss + 
                                                              self.drug_chemical_reconstruct_loss +
                                                              self.drug_disease_reconstruct_loss + 
                                                              self.drug_sideeffect_reconstruct_loss +
                                                              self.protein_protein_reconstruct_loss +
                                                              self.protein_sequence_reconstruct_loss +
                                                              self.protein_disease_reconstruct_loss) + self.l2_loss

    def Embed(self, feat, adj, input_dim):
        if self.version == 'gae':
            embeddings = GraphConvolution(input_dim=input_dim,
                                       output_dim=self.out_dim,
                                       adj=adj,
                                       act=tf.nn.relu,
                                       dropout=self.dropout,
                                       logging=self.logging)(feat)
        elif self.version == 'gvae':
            num_nodes = adj.shape[0]
            z_mean = GraphConvolution(input_dim=input_dim,
                                          output_dim=self.out_dim,
                                          adj=adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                      logging=self.logging)(feat)
            num_nodes = adj.shape[0]
            z_log_std = GraphConvolution(input_dim=input_dim,
                                         output_dim=self.out_dim,
                                         adj=adj,
                                         act=lambda x: x,
                                         dropout=self.dropout,
                                         logging=self.logging)(feat)
            embeddings = z_mean + tf.random_normal([num_nodes, self.out_dim]) * tf.exp(z_log_std)
        else:
            print('Error! No model named %s!'%self.version)
        return embeddings

