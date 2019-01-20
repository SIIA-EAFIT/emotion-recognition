(function () {

    var video = document.getElementById('video'),
        canvas = document.getElementById('canvas'),
        context = canvas.getContext('2d');
    photo = document.getElementById('photo')
    vendorUrl = window.URL || window.webkitURL;

    // const mediaSource = new MediaSource();
    // const video = document.createElement('video');
    // video.srcObject = mediaSource;

    navigator.getMedia = (navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.mozGetUserMedia);

    navigator.getMedia({

        video: true,
        audio: false
    }, function (stream) {
        try {
            video.srcObject = stream;
        } catch (error) {
            video.src = window.URL.createObjectURL(stream);
        }
        video.play();

    }, function (error) {

    });

    document.getElementById('capture').addEventListener('click', function () {
        context.drawImage(video, 0, 0, 400, 300);
        photo.setAttribute('src', canvas.toDataURL('image/png'));
        // create_img();

    });


})();


function send_img() {
    //Make this url, in a different file, so can be accessed from everywhere
    img = document.getElementById('photo');
    emotion = document.getElementById('emotion');
    var emotion_dict = {
        "Angry": "../static/Imagenes/angry.jpg",
        "Disgust": "../static/Imagenes/disgust.png",
        "Fear": "../static/Imagenes/fear.png",
        "Happy": "../static/Imagenes/happy.jpg",
        "Sad": "../static/Imagenes/sad.jpg",
        "Surprise": "../static/Imagenes/surprise.jpg",
        "Neutral": "../static/Imagenes/neutral.png"
    }

    const url = "http://localhost:8000/saveImage";
    fetch(url, {
        method: "POST",
        body: JSON.stringify(img.src),
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(res => res.json())
        .catch(error => console.error('Error:', error))
        .then(response => {
            console.log('Success:', response);
            emotion.src = emotion_dict[response.result];
            var ctx = document.getElementById("myChart").getContext('2d');
            var myChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                    datasets: [{
                        label: 'Probability of feeling each emotion',
                        data: response.prob ,
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.2)',
                            'rgba(54, 162, 235, 0.2)',
                            'rgba(255, 206, 86, 0.2)',
                            'rgba(75, 192, 192, 0.2)',
                            'rgba(153, 102, 255, 0.2)',
                            'rgba(255, 159, 64, 0.2)'
                        ],
                        borderColor: [
                            'rgba(255,99,132,1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)',
                            'rgba(153, 102, 255, 1)',
                            'rgba(255, 159, 64, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        yAxes: [{
                            ticks: {
                                beginAtZero: true
                            }
                        }]
                    }
                }
            });

        });
}

function previewFile() {
    var preview = document.querySelector('img');
    var file = document.querySelector('input[type=file]').files[0];
    var reader = new FileReader();

    reader.onloadend = function () {
        photo.setAttribute('src', reader.result);

    }

    if (file) {
        reader.readAsDataURL(file);
    } else {
        preview.src = "";
    }
}
