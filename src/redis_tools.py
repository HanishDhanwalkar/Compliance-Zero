import redis

r = redis.Redis(host="localhost", port=6379, db=0)

def store_summary(user_id: str, summary: str):
    key = f"user:{user_id}:summaries"
    r.rpush(key, summary)
    r.ltrim(key, -5, -1)   # keep last 5


def load_summaries(user_id: str):
    key = f"user:{user_id}:summaries"
    data = r.lrange(key, -5, -1)
    return [d.decode("utf-8") for d in data]



# docker run -d \
#   --name redis \
#   -p 6379:6379 \
#   redis:7
