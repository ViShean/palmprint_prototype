from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRETE_KEY']="Hello"
    
    # Declare all views/blueprints
    
    from .views import views

    
    app.register_blueprint(views, url_prefix= '/')
    # app.register_blueprint(search, url_prefix="/search")
    
    
    return app