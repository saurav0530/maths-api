import os,process
from urllib import response
from flask import Flask, request, send_file

app = Flask(__name__)
UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def display():
    return "Hello from server..."

@app.route('/upload', methods=['POST','GET'])
def upload_file_post():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'image_13.jpg')
        file1.save(path)
        return "Saved☑"
    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    '''
    
@app.route('/result')
def send_result():
    process.process_image()
    return send_file(os.path.join('.\\upload', 'final.png'))

if __name__ == '__main__':
    app.run()