from model_zoo.din_xdeep_bert import DeepFM
from data_analy.user_interactive import UserInteractiveTool

deep_fm = DeepFM(
    10, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20, UserInteractiveTool.ITEM_EMBEDDING_SIZE], 128, 128,
    embedding_size=40, learning_rate=0.008, use_bert=True, num_attention_heads=2, batch_size=512, weight_decay=1e-5,
    deep_layers_activation='sigmoid')
param_groups = deep_fm.named_parameters()
# for key, value in param_groups:
#     print(key)

# for item in deep_fm.named_children():
#     print(item)
for param in deep_fm.parameters():
    print(type(param.data), param.size())
