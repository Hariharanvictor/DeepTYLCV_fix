import torch
torch.use_deterministic_algorithms(False)

import torch.nn as nn
import torch.nn.functional as F
        
class CONTRA_IL6(nn.Module):
    def __init__(self, 
                 feature_dim_list,
                 max_concatenated_len,
                 n_classes,
                 # Transformer param
                 d_model,
                 n_head,
                 num_transformer_layers,
                 # CNN param
                 kernel_size,
                 stride,
                 dilation,
                 ):
        super(CONTRA_IL6, self).__init__()
        self.num_of_features = len(feature_dim_list)
        self.max_concatenated_len = max_concatenated_len
        self.d_model = d_model
        
        self.projector_list = nn.ModuleList()
        for idx, feature_dim in enumerate(feature_dim_list):
            # add remaining dim to the first feature
            remaining_dim = 0
            if idx == 0:
                remaining_dim = self.d_model % len(feature_dim_list) 
            self.projector_list.append(
                nn.Linear(feature_dim, self.d_model // len(feature_dim_list) + remaining_dim)
            )
            
        self.position_embedding = nn.Embedding(self.max_concatenated_len+1, self.d_model)
        
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=n_head,
                activation='gelu',
                dim_feedforward=self.d_model*4,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        nn.init.xavier_normal_(self.cls_token)
        
        
        self.conv_ms1_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=kernel_size // 2)
        self.conv_ms1_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=kernel_size // 2)
        self.conv_ms2_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=kernel_size // 2)
        self.conv_ms2_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=kernel_size // 2)
        self.conv_ms3_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=kernel_size // 2)
        self.conv_ms3_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=kernel_size // 2)
        # self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(self.d_model, n_classes)
    
    def forward(self, X, masks_X):
        batch_size = X[0].size(0)
        
        projected_X = [
            projector(x) for projector, x in zip(self.projector_list, X)
        ]
        
        concatenated_X = []
        
        concatenated_X = torch.cat(projected_X, dim=-1)
        
        padded_X = torch.zeros(
            (batch_size, self.max_concatenated_len, self.d_model), device=X[0].device
        )
        combined_masks = torch.zeros(
            (batch_size, self.max_concatenated_len+1), device=X[0].device, dtype=torch.long
        )
        pos_ids = torch.zeros((batch_size, self.max_concatenated_len+1), device=X[0].device, dtype=torch.long)
        
        max_len_batches = []
        for batch_idx, emb in enumerate(concatenated_X):
            max_len_batches.append(emb.size(0))
            padded_X[batch_idx, :emb.size(0), :] = emb
            combined_masks[batch_idx, emb.size(0):] = 1
            pos_ids[batch_idx, 1:] = torch.arange(1, self.max_concatenated_len+1)
        
        max_len_batches = torch.tensor(max_len_batches, dtype=torch.long)
        padded_X = torch.cat([self.cls_token.expand(padded_X.size(0), -1, -1), padded_X], dim=1)
        pos_embeds = self.position_embedding(pos_ids)
        
        # Sum pos embeds and segment embeds to X
        padded_X = padded_X + pos_embeds
        
        # Push padded_X through Transformer
        transformer_output = self.transformer_encoder(
            padded_X,
            src_key_padding_mask=combined_masks.bool(),
        )
        
        # split [cls] and the remaining embeddings
        cls_embeds = transformer_output[:, 0, :]
        seq_embeds = transformer_output[:, 1:, :]
        
        seq_embeds = seq_embeds.permute(0, 2, 1)
        seq_embeds = F.relu(self.conv_1(seq_embeds))
        seq_embeds = F.relu(self.conv_2(seq_embeds))
        
        seq_max = self.global_max_pool(seq_embeds)
        seq_avg = self.global_avg_pool(seq_embeds)
        
        seq_outputs = (seq_max + seq_avg) / 2
        seq_outputs = seq_outputs.squeeze(-1)
        
        # final_embeds = seq_outputs + cls_embeds
        
        outputs = self.fc(seq_outputs + cls_embeds) # skip connection
        
        return outputs
    
    
class CONTRA_IL6_CONV(nn.Module):
    def __init__(self, 
                 feature_dim_list,
                 max_concatenated_len,
                 n_classes,
                 # Transformer param
                 d_model,
                 n_head,
                 num_transformer_layers,
                 # CNN param
                 kernel_size,
                 stride,
                 dilation,
                 fc_1,
                 norm_type,
                 dp_size
                 ):
        super(CONTRA_IL6_CONV, self).__init__()
        self.num_of_features = len(feature_dim_list)
        self.max_concatenated_len = max_concatenated_len
        self.d_model = d_model
        self.fc_1_layers=fc_1
        self.norm_type=norm_type
        self.dp_size=dp_size
        
        self.projector_list = nn.ModuleList()
        for idx, feature_dim in enumerate(feature_dim_list):
            # add remaining dim to the first feature
            remaining_dim = 0
            if idx == 0:
                remaining_dim = self.d_model % len(feature_dim_list) 
            self.projector_list.append(
                nn.Linear(feature_dim, self.d_model // len(feature_dim_list) + remaining_dim)
            )
            
        self.position_embedding = nn.Embedding(self.max_concatenated_len+1, self.d_model)
        
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=n_head,
                activation='gelu',
                dim_feedforward=self.d_model*4,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        nn.init.xavier_normal_(self.cls_token)
        
        
        self.conv_ms_1_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size[0], stride=stride, dilation=dilation, padding=kernel_size[0] // 2)
        self.conv_ms_1_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size[0], stride=stride, dilation=dilation, padding=kernel_size[0] // 2)
        self.conv_ms_2_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size[1], stride=stride, dilation=dilation, padding=kernel_size[1] // 2)
        self.conv_ms_2_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size[1], stride=stride, dilation=dilation, padding=kernel_size[1] // 2)
        self.conv_ms_3_1 = nn.Conv1d(self.d_model, self.d_model // 2, kernel_size=kernel_size[2], stride=stride, dilation=dilation, padding=kernel_size[2] // 2)
        self.conv_ms_3_2 = nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=kernel_size[2], stride=stride, dilation=dilation, padding=kernel_size[2] // 2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_1 = nn.Linear(self.d_model + 394, self.fc_1_layers)
        self.fc_2 = nn.Linear(self.fc_1_layers, n_classes)
        self.layer_norm= nn.LayerNorm(self.d_model)
        self.batch_norm=nn.BatchNorm1d(self.d_model)
        self.drop_out=nn.Dropout(self.dp_size)
    
    def forward(self, X, masks_X,X_conv):
        batch_size = X[0].size(0)
        
        projected_X = [
            projector(x) for projector, x in zip(self.projector_list, X)
        ]
        
        concatenated_X = []
        
        concatenated_X = torch.cat(projected_X, dim=-1)
        
        padded_X = torch.zeros(
            (batch_size, self.max_concatenated_len, self.d_model), device=X[0].device
        )
        combined_masks = torch.zeros(
            (batch_size, self.max_concatenated_len+1), device=X[0].device, dtype=torch.long
        )
        pos_ids = torch.zeros((batch_size, self.max_concatenated_len+1), device=X[0].device, dtype=torch.long)
        
        max_len_batches = []
        for batch_idx, emb in enumerate(concatenated_X):
            max_len_batches.append(emb.size(0))
            padded_X[batch_idx, :emb.size(0), :] = emb
            combined_masks[batch_idx, emb.size(0):] = 1
            pos_ids[batch_idx, 1:] = torch.arange(1, self.max_concatenated_len+1)
        
        max_len_batches = torch.tensor(max_len_batches, dtype=torch.long)
        padded_X = torch.cat([self.cls_token.expand(padded_X.size(0), -1, -1), padded_X], dim=1)
        pos_embeds = self.position_embedding(pos_ids)
        
        # Sum pos embeds and segment embeds to X
        padded_X = padded_X + pos_embeds
        
        # Push padded_X through Transformer
        transformer_output = self.transformer_encoder(
            padded_X,
            src_key_padding_mask=combined_masks.bool(),
        )
        
        # split [cls] and the remaining embeddings
        cls_embeds = transformer_output[:, 0, :]
        seq_embeds = transformer_output[:, 1:, :]
        
        seq_embeds = seq_embeds.permute(0, 2, 1)
        seq_embeds_ms_1 = F.leaky_relu(self.conv_ms_1_1(seq_embeds))
        seq_embeds_ms_1 = F.leaky_relu(self.conv_ms_1_2(seq_embeds_ms_1))
        
        seq_embeds_ms_2 = F.leaky_relu(self.conv_ms_2_1(seq_embeds))
        seq_embeds_ms_2 = F.leaky_relu(self.conv_ms_2_2(seq_embeds_ms_2))
        
        seq_embeds_ms_3 = F.leaky_relu(self.conv_ms_3_1(seq_embeds))
        seq_embeds_ms_3 = F.leaky_relu(self.conv_ms_3_2(seq_embeds_ms_3))
        
        # seq_max = self.global_max_pool(seq_embeds)
        seq_avg_1 = self.global_avg_pool(seq_embeds_ms_1)
        seq_avg_2 = self.global_avg_pool(seq_embeds_ms_2)
        seq_avg_3 = self.global_avg_pool(seq_embeds_ms_3)
        # print(seq_avg_1.shape,'1-shape')
        # print(seq_avg_2.shape,'2-shape')
        # print(seq_avg_3.shape,'3-shape')
        
        
        seq_combined = seq_avg_1 + seq_avg_2 + seq_avg_3
        
        # print(seq_combined.shape,'combined_shape')
        seq_combined= seq_combined.squeeze(-1)
        # print(seq_combined.shape,'combined_shape after squeeze')
        if self.norm_type == 'norm':
            seq_outputs=self.layer_norm(seq_combined)
        elif self.norm_type == 'batch':
            seq_outputs=self.batch_norm(seq_combined)
        # seq_outputs = seq_outputs.squeeze(-1)
        
        
        # print(cls_embeds.shape, X_conv.shape, seq_outputs.shape)
        # print(seq_outputs.shape,'embss')
        # print(cls_embeds.shape,'class shape')
        combined_output=seq_outputs + cls_embeds
        # print(combined_output.shape, 'emb class shape')
        # final_embeds = seq_outputs + cls_embeds
        final_output=torch.cat([combined_output,X_conv], dim=-1)
        # print(final_output.shape, 'emb class conv shape')
        pre_output = self.fc_1(final_output) # skip connection
        pre_output=self.drop_out(pre_output)
        outputs=self.fc_2(pre_output)
        
        return outputs
