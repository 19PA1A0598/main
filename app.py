from flask import Flask,render_template,request

import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import os

model1 = load_model('saved_models/model1.hdf5')
model2 = load_model('saved_models/model2.hdf5')
model3 = load_model('saved_models/model3.hdf5')
model4 = load_model('saved_models/model4.hdf5')
model5 = load_model('saved_models/model5.hdf5')
models = [model1,model2,model3,model4,model5]

CATEGORIES = ['1','2','3','4','5','6','7','8','9','10', '11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88']

app = Flask(__name__)

@app.route('/',methods=['GET'])
def hrllo_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile=request.files['imagefile']
    image_path='./static/input.jpg'
    imagefile.save(image_path)

    img=cv2.imread(image_path)
    new_arr=cv2.resize(img,(200,200))
    new_arr=np.array(new_arr)
    new_arr=new_arr.reshape(-1,200,200,3)

    
    preds = [model.predict(new_arr) for model in models]
    preds=np.array(preds)
    summed = np.sum(preds, axis=0)

    prediction1 = model1.predict(new_arr)
    x1=str(CATEGORIES[prediction1.argmax()])
    prediction2 = model2.predict(new_arr)
    x2=str(CATEGORIES[prediction2.argmax()])
    prediction3 = model3.predict(new_arr)
    x3=str(CATEGORIES[prediction3.argmax()])
    prediction4 = model4.predict(new_arr)
    x4=str(CATEGORIES[prediction4.argmax()])
    prediction5 = model5.predict(new_arr)
    x5=str(CATEGORIES[prediction5.argmax()])

    ensemble_prediction = np.argmax(summed, axis=1)
    x=str(ensemble_prediction[0]+1)

    pa='./static/pics/'+x
    files = os.listdir(pa)
    image_file = [file for file in files if file.lower().endswith(('jpg', 'jpeg', 'png'))][0]
    imagepath = os.path.join(pa, image_file)
    img = Image.open(imagepath)
    ou_path='./static/output.jpg'
    img.save(ou_path)

    pa1='./static/pics/'+x1
    files1 = os.listdir(pa1)
    image_file1 = [file for file in files1 if file.lower().endswith(('jpg', 'jpeg', 'png'))][0]
    imagepath1 = os.path.join(pa1, image_file1)
    img1 = Image.open(imagepath1)
    ou_path1='./static/output1.jpg'
    img1.save(ou_path1)

    pa2='./static/pics/'+x2
    files2 = os.listdir(pa2)
    image_file2 = [file for file in files2 if file.lower().endswith(('jpg', 'jpeg', 'png'))][0]
    imagepath2 = os.path.join(pa2, image_file2)
    img2 = Image.open(imagepath2)
    ou_path2='./static/output2.jpg'
    img2.save(ou_path2)

    pa3='./static/pics/'+x3
    files3 = os.listdir(pa3)
    image_file3 = [file for file in files3 if file.lower().endswith(('jpg', 'jpeg', 'png'))][0]
    imagepath3 = os.path.join(pa3, image_file3)
    img3 = Image.open(imagepath3)
    ou_path3='./static/output3.jpg'
    img3.save(ou_path3)

    pa4='./static/pics/'+x4
    files4 = os.listdir(pa4)
    image_file4 = [file for file in files4 if file.lower().endswith(('jpg', 'jpeg', 'png'))][0]
    imagepath4 = os.path.join(pa4, image_file4)
    img4 = Image.open(imagepath4)
    ou_path4='./static/output4.jpg'
    img4.save(ou_path4)

    pa5='./static/pics/'+x5
    files5 = os.listdir(pa5)
    image_file5 = [file for file in files5 if file.lower().endswith(('jpg', 'jpeg', 'png'))][0]
    imagepath5 = os.path.join(pa5, image_file5)
    img5 = Image.open(imagepath5)
    ou_path5='./static/output5.jpg'
    img5.save(ou_path5)


    return render_template('output.html')



if __name__=='__main__':
    app.run(port=6300)