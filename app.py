from flask import Flask, render_template
from datetime import datetime

app = Flask(__name__)

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/node1")
def node1():
    return render_template("node1.html")

@app.route("/node2")
def node2():
    return render_template("node2.html")

@app.route("/combined")
def combined():
    return render_template("combined.html")

if __name__ == "__main__":
    app.run(debug=True)
