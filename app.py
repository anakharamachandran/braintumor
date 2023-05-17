from flask import Flask,render_template,request
app = Flask(__name__)
import json
import classifier as cl
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route("/upload")
def upload():
    return render_template('upload.html')
@app.route('/result', methods=['GET','POST'])
def result():

    if request.method == 'POST':

        image = request.files["pic"]
        image.save("static/uploads/"+image.filename)
        result,pre_img = cl.classify(image.filename)
       
        labels = ["glioma","meningioma","notumor","pituitary"]
        ans = labels[result]
        
        return render_template('result.html', ans = ans,imagepath=image.filename,pre_img=pre_img)

    else:

        return render_template('result.html', name="shine")