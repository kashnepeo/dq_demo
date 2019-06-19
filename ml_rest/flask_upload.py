import csv, os
from flask import Flask, render_template, request , redirect , url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/upload')
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload' , methods = ['GET' , 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        f = open(secure_filename(f.filename))
        lists = csv.reader(f)
        resultList = []
        for list in lists:
            resultList.append(list)
        f.close
        return render_template('grid/grid.html',resultList=resultList)

if __name__ == '__main__':
    app.run(debug=True)