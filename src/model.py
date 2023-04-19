

class SimpleTwoTower(nn.Module):
    
    def __init__(self, n_items, n_users, embedding_size, ln=None):
        super(SimpleTwoTower, self).__init__()


        self.userid_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
        ])

        self.user_id_embedding = nn.Sequential([
            
            nn.Embedding(len(unique_user_ids), 5, padding_idx=0),
            nn.Embedding(num_embeddings=(n_users + 1), embedding_dim=embedding_size)
        ])

        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
        ])
        self.normalized_timestamp = tf.keras.layers.Normalization(
            axis=None
        )

        self.normalized_timestamp.adapt(timestamps)
        # --------------------------------------------------------------
        
        # self.ln = ln
        self.item_emb = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_size)  # self.ln[0]
        # We add +1 additional embedding to account for unknown tokens.
        self.user_emb = nn.Embedding(num_embeddings=n_users + 1, embedding_dim=embedding_size)
       
        
        self.item_layers = [] #nn.ModuleList()
        self.user_layers = [] #nn.ModuleList()
        
        # for i, n in enumerate(ln[0:-1]):
        #     m = int(ln[i+1])
        self.item_layers.append(nn.Linear(embedding_size, embedding_size, bias=True))  # n, m
        self.item_layers.append(nn.ReLU())
        
        self.user_layers.append(nn.Linear(embedding_size, embedding_size, bias=True))
        self.user_layers.append(nn.ReLU())   # is this ReLU needed???
            
            
        self.item_layers = nn.Sequential(*self.item_layers)
        self.user_layers = nn.Sequential(*self.user_layers)
        
        self.dot = torch.matmul
        self.sigmoid = nn.Sigmoid()

    def forward(self, items, users):

        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.timestamp_embedding(inputs["timestamp"]),
            tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
        ], axis=1)

        
        item_emb = self.item_emb(items)
        user_emb = self.user_emb(users)
        
        item_emb = self.item_layers(item_emb)
        user_emb = self.user_layers(user_emb)

        print(user_emb.shape, item_emb.shape)
        dp = self.dot(user_emb, torch.permute(item_emb, (0, 2, 1)))
        dp = dp.sum(dim=1).squeeze()

        return self.sigmoid(dp)