import base64, os,process
import threading
from urllib import response
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        
@app.route('/')
def display():
    return "Hello from server..."

@app.route('/upload', methods=['POST','GET'])
def upload_files():
    if request.method == 'POST':
        if 'image' not in request.form:
            return jsonify(
                response = 'there is no image in form!'
            )
        
        file = request.form['image']
        autocrop = request.form['autocrop']
        print(autocrop)
        imgdata = base64.b64decode(file)
        filename = 'upload/input.jpg'  
        with open(filename, 'wb') as f:
            f.write(imgdata)

        return jsonify(
                response = 'Uploaded...'
            )
    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image">
      <input type="submit">
    </form>
    '''
    
@app.route('/result')
def send_result():
    filepath = ".\\upload\\final.png"
    if(os.path.exists(filepath)):
        os.remove(filepath)

    t = threading.Thread(target=process.process_image)
    t.setDaemon(True)
    t.start()
    t.join()
    
    if(os.path.exists(filepath)):
        return send_file(os.path.join('.\\upload', 'final.png'))
    return send_file(os.path.join('.\\upload', 'error.png'))

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)