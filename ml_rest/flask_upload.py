import csv
from flask import Flask, render_template, request , redirect , url_for
from werkzeug.utils import secure_filename
import timeit

app = Flask(__name__)


@app.route('/upload')
def render_file():
    return render_template('grid/upload.html')


@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        f = open(secure_filename(f.filename))
        lists = csv.reader(f)
        resultList = []
        for list in lists:
            resultList.append([except_fn(x) for x in list])
        f.close
        return render_template('grid/grid.html', resultList=resultList)


def except_fn(x):
    try:
        return "{:d}".format(round(float(x)))
    except ValueError:
        return x


if __name__ == '__main__':
    app.run(debug=True)
