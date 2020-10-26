from camera import Camera
from flask import Flask, render_template, Response

# Initialize flask app
app = Flask(__name__, template_folder='templates')


# Rout path for home
@app.route('/')
def index():
    return render_template("index.html")


def render_frame(camera):
    while True:
        frame = camera.get_frames()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Rout path to feed the video
@app.route('/video')
def video():
    return Response(
        render_frame(Camera()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(debug=True)
