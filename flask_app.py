from flask import Flask

app = Flask(__name__)

@app.route('/')
def welcome():
    """ This endpoint should show the intent of the web page.

    It should also show some buttons to navigate to the about and the
    infer section of the page.

    """
    return "This app will output the label for the emotion of your face"
    

@app.route('/about')
def about():
    return 'Thanks for visiting us'

@app.rout('/infer/')
def infer():
    """ Estimates the emotion in a given face.

    This endpoint should use the user's computer camera to output the 
    corresponding emotion that is presented to it.

    """
    raise NotImplementedError



if __name__ == '__main__':
    app.run()