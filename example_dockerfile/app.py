from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, Docker!'


if __name__ == '__main__':
    import os
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
