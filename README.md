# Compliance-Zero



RUN redis:
```
docker run -d --name redis -p 6379:6379 redis:7
```
test:
```
docker exec -it redis redis-cli ping
```