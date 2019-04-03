from flask import Flask, jsonify
import json
import random
from  gevent.pywsgi import WSGIServer
from gevent import monkey
import logging
monkey.patch_all()


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

video_dict = dict()

for i in range(10000):
    video_dict[i] = json.dumps([random.random() for _ in range(128)])


@app.route('/video/<video_id>')
def hello_world(video_id):
    return video_dict[int(video_id)]
    # return "ok"


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

