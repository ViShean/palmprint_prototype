from flask import Blueprint,render_template,request,redirect,url_for,jsonify
import random
from datetime import datetime 



views = Blueprint('views',__name__,static_folder='static')

# This file is a blueprint that has lots of urls, routes!
# Each route has a function which is for this is each view's function


@views.route('/')
def home():
    # now = datetime.now()
    # formatted_date_time = now.strftime("%a, %b %d %I:%M %p")
    return render_template("main.html")

@views.route('/collection')
def collect():
    return render_template("search.html")


@views.route('/validation')
def validate():
   return render_template("validation.html")


