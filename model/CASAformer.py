import torch.nn as nn
import torch
import time
# from torchinfo import summary


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        
        out = self.out_proj(out)

        return out, attn_score


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out, attn_score = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out, attn_score


class LSHAttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, bucket_num=5, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.bucket_num = bucket_num
        self.bucket_size = 0

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)
        
    def LSH_mask(self, x, target_length, congest_speed_threshold=-0.2260138304488262, zero=-2.7909362773165847): 
    # current threshold is 50 mph -0.2260138304488262 (30 mph (-1.2519828091959295)) -2.7909362773165847->0 for METRLA; -0.28984798470914763 for PEMSBAY 60 mph -6.647014197551572->0
        # x (batch_size, n_nodes, seq_len)
        batch_size, n_nodes, seq_len = x.size()
        # x = x.unsqueeze(-1).reshape(batch_size, seq_len, n_nodes, 1)
        # x_mean = torch.mean(x, -1)
        # x_mean = x_mean.unsqueeze(-1)
        dist_matrix = torch.cdist(x, x, p=1)
        # print(dist_matrix.size())
        # batch_size, seq_len, num_nodes, num_nodes
        
        q1 = torch.quantile(dist_matrix, 0.01, -1)
        q21 = torch.quantile(dist_matrix, 0.1, -1)
        q22 = torch.quantile(dist_matrix, 0.11, -1)
        q31 = torch.quantile(dist_matrix, 0.2, -1)
        q32 = torch.quantile(dist_matrix, 0.21, -1)
        q41 = torch.quantile(dist_matrix, 0.3, -1)
        q42 = torch.quantile(dist_matrix, 0.31, -1)
        q51 = torch.quantile(dist_matrix, 0.4, -1)
        q52 = torch.quantile(dist_matrix, 0.41, -1)
        q61 = torch.quantile(dist_matrix, 0.5, -1)
        q62 = torch.quantile(dist_matrix, 0.51, -1)
        q71 = torch.quantile(dist_matrix, 0.6, -1)
        q72 = torch.quantile(dist_matrix, 0.61, -1)
        q81 = torch.quantile(dist_matrix, 0.7, -1)
        q82 = torch.quantile(dist_matrix, 0.71, -1)
        q91 = torch.quantile(dist_matrix, 0.8, -1)
        q92 = torch.quantile(dist_matrix, 0.81, -1)
        q101 = torch.quantile(dist_matrix, 0.9, -1)
        q102 = torch.quantile(dist_matrix, 0.91, -1)
        
        q1 = q1.unsqueeze(-1).repeat(1, 1, n_nodes)
        q21 = q21.unsqueeze(-1).repeat(1, 1, n_nodes)
        q22 = q22.unsqueeze(-1).repeat(1, 1, n_nodes)
        q31 = q31.unsqueeze(-1).repeat(1, 1, n_nodes)
        q32 = q32.unsqueeze(-1).repeat(1, 1, n_nodes)
        q41 = q41.unsqueeze(-1).repeat(1, 1, n_nodes)
        q42 = q42.unsqueeze(-1).repeat(1, 1, n_nodes)
        q51 = q51.unsqueeze(-1).repeat(1, 1, n_nodes)
        q52 = q52.unsqueeze(-1).repeat(1, 1, n_nodes)
        q61 = q61.unsqueeze(-1).repeat(1, 1, n_nodes)
        q62 = q62.unsqueeze(-1).repeat(1, 1, n_nodes)
        q71 = q71.unsqueeze(-1).repeat(1, 1, n_nodes)
        q72 = q72.unsqueeze(-1).repeat(1, 1, n_nodes)
        q81 = q81.unsqueeze(-1).repeat(1, 1, n_nodes)
        q82 = q82.unsqueeze(-1).repeat(1, 1, n_nodes)
        q91 = q91.unsqueeze(-1).repeat(1, 1, n_nodes)
        q92 = q92.unsqueeze(-1).repeat(1, 1, n_nodes)
        q101 = q101.unsqueeze(-1).repeat(1, 1, n_nodes)
        q102 = q102.unsqueeze(-1).repeat(1, 1, n_nodes)
        
        min_mask = torch.le(dist_matrix, q1)
        mask_2 = torch.logical_and(torch.ge(dist_matrix, q21), torch.le(dist_matrix, q22))
        mask_3 = torch.logical_and(torch.ge(dist_matrix, q31), torch.le(dist_matrix, q32))
        mask_4 = torch.logical_and(torch.ge(dist_matrix, q41), torch.le(dist_matrix, q42))
        mask_5 = torch.logical_and(torch.ge(dist_matrix, q51), torch.le(dist_matrix, q52))
        mask_6 = torch.logical_and(torch.ge(dist_matrix, q61), torch.le(dist_matrix, q62))
        mask_7 = torch.logical_and(torch.ge(dist_matrix, q71), torch.le(dist_matrix, q72))
        mask_8 = torch.logical_and(torch.ge(dist_matrix, q81), torch.le(dist_matrix, q82))
        mask_9 = torch.logical_and(torch.ge(dist_matrix, q91), torch.le(dist_matrix, q92))
        mask_10 = torch.logical_and(torch.ge(dist_matrix, q101), torch.le(dist_matrix, q102))
        
        mask = torch.logical_or(min_mask, mask_2)
        mask = torch.logical_or(mask, mask_3)
        mask = torch.logical_or(mask, mask_4)
        mask = torch.logical_or(mask, mask_5)
        mask = torch.logical_or(mask, mask_6)
        mask = torch.logical_or(mask, mask_7)
        mask = torch.logical_or(mask, mask_8)
        mask = torch.logical_or(mask, mask_9)
        mask = torch.logical_or(mask, mask_10)
        
        x_mean = torch.mean(x, -1)
        x_mean = x_mean.reshape(batch_size, 1, n_nodes)
        x_mean = x_mean.repeat(1, n_nodes, 1)
        
        threshold = torch.ones(batch_size, n_nodes, n_nodes, device=x.device)
        congest_mask = torch.le(x_mean, threshold * congest_speed_threshold)
        
        mask = torch.logical_or(mask, congest_mask)
        mask = torch.cat((mask, torch.ones((batch_size, n_nodes, target_length - n_nodes), dtype=torch.bool, device=mask.device)), -1)
        mask = mask.reshape(batch_size, 1, n_nodes, target_length)
        
        mask = mask.repeat(self.num_heads, seq_len, 1, 1)
        
        
        return mask 
    
    def balance_congest_nodes(self, key, value, x, SCALER):
        original_x = SCALER.inverse_transform(x)
        x_mean = torch.mean(original_x, -1)
        batch_size, n_nodes, seq_len = original_x.size()
        
        max_len = 0
        new_key = []
        new_value = []
        # new_x = []
        for i in range(batch_size):
            new_key.append(key[i])
            new_value.append(value[i])
            # new_x.append(x[i])
        
        for threshold in range(20, 60, 10):
            if threshold == 20:
                congest_idx = torch.argwhere(torch.logical_and(0 < x_mean, x_mean <= threshold))
            else:
                congest_idx = torch.argwhere(torch.logical_and(threshold - 10 < x_mean, x_mean <= threshold))
                
            for i in range(len(congest_idx)):
                batch_idx = congest_idx[i][0]
                node_idx = congest_idx[i][1]
                
                resampled_key = key[batch_idx, :, node_idx, :].unsqueeze(1).repeat(1, int(1 / threshold * 300), 1)
                resampled_value = value[batch_idx, :, node_idx, :].unsqueeze(1).repeat(1, int(1 / threshold * 300), 1)
                # resampled_x = x[batch_idx, node_idx, :].unsqueeze(0).repeat(int(1 / threshold * 300), 1)
                
                new_key[batch_idx] = torch.cat((new_key[batch_idx], resampled_key), -2)
                new_value[batch_idx] = torch.cat((new_value[batch_idx], resampled_value), -2)
                # new_x[batch_idx] = torch.cat((new_x[batch_idx], resampled_x), -2)
            
        # max_l = 0
        # for i in range(batch_size):
        #     print(i, new_key[i].size(), new_value[i].size(), new_x[i].size())
        
        return new_key, new_value

    def forward(self, query, key, value, x, distance_matrix, SCALER, resample_congest_node=False):
        # Q    (batch_size, ..., n_nodes, model_dim)
        # K, V (batch_size, ..., n_nodes, model_dim)
        # x    (batch_size, n_nodes, seq_len)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        seq_len = x.shape[-1]
        # print(x.shape)
        
        if not resample_congest_node:
            query = self.FC_Q(query)
            key = self.FC_K(key)
            value = self.FC_V(value)
            
            target_length = key.size(-2)

            # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
            query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
            key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
            value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        
            key = key.transpose(
                -1, -2
            )  # (num_heads * batch_size, ..., head_dim, src_length)
        
            attn_score = (
                query @ key
            ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)
        
            mask = self.LSH_mask(x, target_length)
            
            # if not distance_matrix is None:
            #     attn_score = attn_score + distance_matrix
        
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place
            
            attn_score = torch.softmax(attn_score, dim=-1)
        
            out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
            out = torch.cat(
                torch.split(out, batch_size, dim=0), dim=-1
            )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        
            out = self.out_proj(out)
        
        # repeat congest key and value
        else:
            new_key, new_value = self.balance_congest_nodes(key, value, x, SCALER)
            new_query = query
            new_x = x
            for i in range(batch_size):
                query = new_query[i].unsqueeze(0)
                key = new_key[i].unsqueeze(0)
                value = new_value[i].unsqueeze(0)
                x = new_x[i].unsqueeze(0)

                query = self.FC_Q(query)
                key = self.FC_K(key)
                value = self.FC_V(value)
            
                target_length = key.size(-2)

                # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
                query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
                key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
                value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        
                key = key.transpose(
                    -1, -2
                )  # (num_heads * batch_size, ..., head_dim, src_length)
         
                attn_score = (
                    query @ key
                ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)
        
                mask = self.LSH_mask(x, target_length)
        
                attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

                attn_score = torch.softmax(attn_score, dim=-1)
        
                out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
                out = torch.cat(
                    torch.split(out, 1, dim=0), dim=-1
                ) 
        
                out = self.out_proj(out)

        return out, attn_score


class SparseAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, bucket_num=5, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = LSHAttentionLayer(model_dim, bucket_num, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, orginal_x, distance_matrix, SCALER, dim=-2):
        real_num_node = orginal_x.size(1)
        query = x[:, :, 0:real_num_node, :]
        query = query.transpose(dim, -2)
        key = x.transpose(dim, -2)
        value = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = query
        out, attn_score = self.attn(query, key, value, orginal_x, distance_matrix, SCALER)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out, attn_score
        
        
class mySTAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        bucket_num=5,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        virtual_nodes=False,
        use_distance_matrix=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.virtual_nodes = virtual_nodes
        self.use_distance_matrix = use_distance_matrix

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            if not virtual_nodes:
                self.adaptive_embedding = nn.init.xavier_uniform_(
                    nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
                )
            else:
                self.adaptive_embedding = nn.init.xavier_uniform_(
                    nn.Parameter(torch.empty(in_steps, 270, adaptive_embedding_dim))
                )
        
        if use_distance_matrix:
            self.adj_matrix_encoder = nn.Conv2d(
                1, out_steps, 1
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                # SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                SparseAttentionLayer(self.model_dim, feed_forward_dim, bucket_num, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, distance_matrix, SCALER):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size, in_steps, num_nodes, _ =  x.size()
        batch_size = x.shape[0]
        original_x = x[..., 0].transpose(-1, -2)
        
        if not self.virtual_nodes:
            if self.tod_embedding_dim > 0:
                tod = x[..., 1]
            if self.dow_embedding_dim > 0:
                dow = x[..., 2]
            x = x[..., : self.input_dim]
            x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        else:
            if self.tod_embedding_dim > 0:
                tod = x[..., 1]
            if self.dow_embedding_dim > 0:
                dow = x[..., 2]
            x = x[..., : self.input_dim]
            
            for threshold in range(20, 60, 10):
                if threshold < 40:
                    # new_added_x = torch.rand((batch_size, in_steps, int( 1 / threshold * 500), 1), device=x.device) * threshold
                    new_added_x = torch.randn((batch_size, in_steps, int( 1 / threshold * 600), 1), device=x.device) * threshold
                else:
                    # new_added_x = torch.rand((batch_size, in_steps, int( 1 / threshold * 500), 1), device=x.device) * 10 + threshold
                    new_added_x = torch.randn((batch_size, in_steps, int( 1 / threshold * 300), 1), device=x.device) * threshold
                new_added_tod = tod[:, :, 0: new_added_x.size(2)]
                new_added_dow = dow[:, :, 0: new_added_x.size(2)]
                
                tod = torch.cat((tod, new_added_tod), -1)
                dow = torch.cat((dow, new_added_dow), -1)
                
                new_added_x = torch.cat((new_added_x, new_added_tod.unsqueeze(-1), new_added_dow.unsqueeze(-1)), -1)
                x = torch.cat((x, new_added_x), -2)
            x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)    
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        # if self.spatial_embedding_dim > 0:
        #     spatial_emb = self.node_emb.expand(
        #         batch_size, self.in_steps, *self.node_emb.shape
        #     )
        #     features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)   
        
        """
        if self.use_distance_matrix:
            if self.virtual_nodes:
                new_added_distance_matrix = torch.ones((self.num_nodes, x.size(-2) - self.num_nodes), device=x.device)
                distance_matrix = torch.cat((distance_matrix, new_added_distance_matrix), -1)
                enc_distance_matrix = self.adj_matrix_encoder(distance_matrix.reshape((1, self.num_nodes, x.size(-2))))
                enc_distance_matrix = torch.sigmoid(enc_distance_matrix)
            else:
                enc_distance_matrix = self.adj_matrix_encoder(distance_matrix.reshape((1, self.num_nodes, self.num_nodes)))
                enc_distance_matrix = torch.sigmoid(enc_distance_matrix)
        """

        attn_scores = []
        for attn in self.attn_layers_t:
            x, attn_score_t = attn(x, dim=1)
        for attn in self.attn_layers_s:
            new_added_input_embedding = x[:, :, self.num_nodes::, :]
            # if self.use_distance_matrix:
            #     x, attn_score_s = attn(x, original_x, enc_distance_matrix, SCALER, dim=2)
            # else:
            x, attn_score_s = attn(x, original_x, None, SCALER, dim=2)
            if self.virtual_nodes:
                x = torch.cat((x, new_added_input_embedding), -2)
            # x, attn_score_s = attn(x, dim=2)
            # if len(attn_scores) == 0:
            #     attn_scores = attn_score_s
            # else:
            #     attn_scores += attn_score_s
        # attn_scores /= len(self.attn_layers_s)
        x = x[:, :, 0: self.num_nodes, :]
            
        # (batch_size, in_steps, num_nodes, model_dim)
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)
        
        return out #, attn_scores


if __name__ == "__main__":
    model = STAEformer(207, 12, 12)
    # summary(model, [64, 12, 207, 3])
