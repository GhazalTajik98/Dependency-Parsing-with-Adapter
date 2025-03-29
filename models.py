import torch
from transformers import XLMRobertaModel
from config import BASE_MODEL_NAME, DEVICE, RELATION_NUM


class InitialModel(torch.nn.Module):
    def __init__(self, HIDDEN_DIM, OUTPUT_DIM, num_relations):
        super().__init__()

        # Load pretrained XLM-RoBERTa model
        self.xlm_roberta = XLMRobertaModel.from_pretrained(BASE_MODEL_NAME)

        # Define linear layers for head and dependent token representations (for arcs)
        self.h_head = torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)  # For arc heads
        self.relu1 = torch.nn.ReLU()
        self.h_dep = torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)   # For arc dependents
        self.relu2 = torch.nn.ReLU()

        # Define parameters for the bilinear scoring function (for arcs)
        self.U1 = torch.nn.Parameter(torch.empty(OUTPUT_DIM, OUTPUT_DIM))
        self.u2 = torch.nn.Parameter(torch.empty(OUTPUT_DIM))

        # Define linear layers for relation prediction
        self.OUTPUT_DIM_REL = 100  # Smaller dimension for relation representations
        self.h_head_rel = torch.nn.Linear(HIDDEN_DIM, self.OUTPUT_DIM_REL)  # For relation heads
        self.h_dep_rel = torch.nn.Linear(HIDDEN_DIM, self.OUTPUT_DIM_REL)   # For relation dependents
        self.relu_head_rel = torch.nn.ReLU()
        self.relu_dep_rel = torch.nn.ReLU()

        # Define biaffine parameter for relation scoring
        self.num_relations = num_relations
        self.U_rel = torch.nn.Parameter(
            torch.empty(self.OUTPUT_DIM_REL, self.num_relations, self.OUTPUT_DIM_REL)
        )

        # Initialize parameters
        torch.nn.init.xavier_uniform_(self.U1)
        torch.nn.init.zeros_(self.u2)
        torch.nn.init.xavier_uniform_(self.U_rel)

        # Freeze all layers of XLM-RoBERTa by default
        for param in self.xlm_roberta.parameters():
            param.requires_grad = False

        # Unfreeze the last N layers of XLM-RoBERTa
        N = 4
        total_layers = len(self.xlm_roberta.encoder.layer)
        for i in range(total_layers - N, total_layers):
            for param in self.xlm_roberta.encoder.layer[i].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # Forward pass through XLM-RoBERTa
        roberta_output = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_layer = roberta_output.last_hidden_state  # Shape: (batch_size, seq_len, HIDDEN_DIM)

        # Compute representations for arc prediction
        H_head = self.h_head(hidden_layer)      # Shape: (batch_size, seq_len, OUTPUT_DIM)
        head_relu = self.relu1(H_head)
        H_dep = self.h_dep(hidden_layer)        # Shape: (batch_size, seq_len, OUTPUT_DIM)
        dep_relu = self.relu2(H_dep)
        # Compute arc scores
        score = torch.matmul(
            torch.matmul(head_relu, self.U1), dep_relu.transpose(1, 2)
        )  # Shape: (batch_size, seq_len, seq_len)
        score += torch.matmul(head_relu, self.u2).unsqueeze(-1)
        # Compute representations for relation prediction
        H_head_rel = self.h_head_rel(hidden_layer)      # Shape: (batch_size, seq_len, OUTPUT_DIM_REL)
        H_head_rel = self.relu_head_rel(H_head_rel)
        H_dep_rel = self.h_dep_rel(hidden_layer)        # Shape: (batch_size, seq_len, OUTPUT_DIM_REL)
        H_dep_rel = self.relu_dep_rel(H_dep_rel)

        relation_scores = torch.einsum('bid,drf,bjf->bijr', H_head_rel, self.U_rel, H_dep_rel)

        # Return both arc and relation scores
        return {'arc_scores': score, 'rel_scores': relation_scores}




class CustomAdapter(torch.nn.Module):
    def __init__(self, hidden_dim, adapter_dim=64):
        super().__init__()
        self.down = torch.nn.Linear(hidden_dim, adapter_dim)
        self.relu = torch.nn.ReLU()
        self.up = torch.nn.Linear(adapter_dim, hidden_dim)


    def forward(self, x):
        return self.up(self.relu(self.down(x))) + x



class AdapterRobertaOutput(torch.nn.Module):
    def __init__(self, original_output, adapter_dim=64):
        super().__init__()
        self.dense = original_output.dense  # FFN’s linear projection
        self.LayerNorm = original_output.LayerNorm
        self.dropout = original_output.dropout
        # Add the adapter after dense, using the hidden dimension (e.g., 768 for XLM-RoBERTa base)
        self.adapter = CustomAdapter(original_output.dense.out_features, adapter_dim)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # FFN projection
        hidden_states = self.adapter(hidden_states)  # Apply adapter
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # Residual + norm
        return hidden_states

class ExtendedModelWithPfeiffer(torch.nn.Module):
    def __init__(self, initial_model, adapter_dim=64):
        super().__init__()
        # Store the initial model (trained on English)
        self.initial_model = initial_model
        
        # Add Pfeiffer adapters to each RoBERTa layer
        for layer in self.initial_model.xlm_roberta.encoder.layer:
            original_output = layer.output
            layer.output = AdapterRobertaOutput(original_output, adapter_dim)
        
        # Freeze all parameters in the initial model
        for param in self.initial_model.parameters():
            param.requires_grad = False
        
        # Enable training only for adapter parameters
        for layer in self.initial_model.xlm_roberta.encoder.layer:
            for param in layer.output.adapter.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # Use the initial model’s forward pass with modified RoBERTa layers
        return self.initial_model(input_ids, attention_mask)


class HoulsbyAttention(torch.nn.Module):
    def __init__(self, original_attention, adapter_dim=64):
        super().__init__()
        self.original_attention = original_attention
        # Adapter matches the output dimension of the attention block
        self.adapter = CustomAdapter(original_attention.output.dense.out_features, adapter_dim)


    def forward(self, hidden_states, attention_mask=None, head_mask=None, 
                encoder_hidden_states=None, encoder_attention_mask=None, 
                past_key_value=None, output_attentions=False):
        # Run the original attention forward pass
        attention_output = self.original_attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions
        )
        # Apply adapter with residual connection to the attention output
        adapter_output = self.adapter(attention_output[0]) + attention_output[0]
        # Preserve additional outputs (e.g., attention weights)
        attention_output = (adapter_output,) + attention_output[1:]
        return attention_output


class HoulsbyOutput(torch.nn.Module):
    def __init__(self, original_output, adapter_dim=64):
        super().__init__()
        self.original_output = original_output
        # Adapter matches the output dimension of the FFN block
        self.adapter = CustomAdapter(original_output.dense.out_features, adapter_dim)

    def forward(self, hidden_states, input_tensor):
        # Run the original FFN forward pass
        ffn_output = self.original_output(hidden_states, input_tensor)
        # Apply adapter with residual connection
        adapter_output = self.adapter(ffn_output) + ffn_output
        return adapter_output

class ExtendedModelWithHoulsby(torch.nn.Module):
    def __init__(self, initial_model, adapter_dim=64):
        super().__init__()
        # Store the initial model (e.g., a trained RoBERTa-based model)
        self.initial_model = initial_model

        # Modify each transformer layer to include Houlsby adapters
        for layer in self.initial_model.xlm_roberta.encoder.layer:
            # Replace the attention module
            original_attention = layer.attention
            layer.attention = HoulsbyAttention(original_attention, adapter_dim)
            # Replace the output (FFN) module
            original_output = layer.output
            layer.output = HoulsbyOutput(original_output, adapter_dim)

        # Freeze all original model parameters
        for param in self.initial_model.parameters():
            param.requires_grad = False

        # Enable training only for adapter parameters
        for layer in self.initial_model.xlm_roberta.encoder.layer:
            for param in layer.attention.adapter.parameters():
                param.requires_grad = True
            for param in layer.output.adapter.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # Forward pass using the modified transformer layers
        return self.initial_model(input_ids, attention_mask)



def model_initializing(model_name, hidden_dim=768, output_dim=256, relation_num=RELATION_NUM, trained_base_model=None):
    match model_name:
        case "base":
            model = InitialModel(hidden_dim, output_dim, relation_num)

        case "pfeiffer":
            model = ExtendedModelWithPfeiffer(trained_base_model, adapter_dim=64)

        case "hously":
            model = ExtendedModelWithHoulsby(trained_base_model, adapter_dim=64)
        
    model.to(DEVICE)
    return model