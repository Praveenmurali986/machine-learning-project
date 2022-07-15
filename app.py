from flask import Flask
from housing.logger import logging
from housing.exception import HousingException
import sys
from flask import abort, send_file, render_template

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        raise HousingException(e,sys) from e

if __name__=='__main__':
    app.run(debug=True)
