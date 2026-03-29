import torch
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

# Load model
model_path = "/your/path/to/embodied-eval-main/embodied_eval/data/wall-oss-fast"
model = Qwen2_5_VLMoEForAction.from_pretrained(model_path)
model.eval()

# Setup
batch_size = 1
seq_length = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).bfloat16()

# Prepare inputs (example with synthetic data)
torch.manual_seed(0)
input_ids = torch.randint(0, len(model.processor.tokenizer), (batch_size, seq_length), dtype=torch.long)
attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
moe_token_types = torch.zeros((batch_size, seq_length), dtype=torch.long)
position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

# Robotics-specific inputs
proprioception = torch.randn((batch_size, 1, 20), dtype=torch.float32)  # Joint states
agent_pos_mask = torch.ones((batch_size, 1, 20), dtype=torch.float32)
dof_mask = torch.ones((batch_size, 32, 20), dtype=torch.float32)  # DOF mask
dataset_names = ["x2_normal"]

# Move to device
inputs = {
    "input_ids": input_ids.to(device),
    "attention_mask": attention_mask.to(device),
    "moe_token_types": moe_token_types.to(device),
    "position_ids": position_ids.to(device),
    "proprioception": proprioception.to(device).bfloat16(),
    "agent_pos_mask": agent_pos_mask.to(device).bfloat16(),
    "dof_mask": dof_mask.to(device).bfloat16(),
    "dataset_names": dataset_names,
    "mode": "validate"
}

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    print(f"Output logits shape: {outputs.logits.shape}")