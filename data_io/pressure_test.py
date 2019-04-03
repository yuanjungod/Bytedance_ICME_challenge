import requests
import time


start = time.time()
count = 0
for _ in range(1000):
    url = "http://localhost:5000/video/%s" % (count % 2000)
    # url = "http://localhost:5000/FQA/%s" % json.dumps(q)
    a = requests.get(url).text
    count += 1
print(time.time() - start)

