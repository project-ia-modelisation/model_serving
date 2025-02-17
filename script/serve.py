import redis
import torch
import io

redis_client = redis.Redis(host="redis", port=6379)

# Récupération du modèle
def load_model_from_redis(key="trained_model"):
    buffer = io.BytesIO(redis_client.get(key))
    model = torch.nn.Linear(10, 1)
    model.load_state_dict(torch.load(buffer))
    return model

model = load_model_from_redis()
