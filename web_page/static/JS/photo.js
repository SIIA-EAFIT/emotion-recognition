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

        video.src = URL.createObjectURL(stream); //Change this line, is deprecated;
        video.play();

    }, function (error) {

    });

    document.getElementById('capture').addEventListener('click', function () {
        context.drawImage(video, 0, 0, 400, 300);
        photo.setAttribute('src', canvas.toDataURL('image/png'));
        create_img();

    });

    function create_img() {
        var myImg = document.createElement("IMG");
        myImg.src = canvas.toDataURL("image/png"); // this will generate base64 data
        send_img(myImg.src);
        // document.getElementById("img_dispplay").innerHTML = "<img src='"+x.src+"' width='400' height='300' class='img-responsive'>";
        // document.body.appendChild(x);
        //console.log(img.src);
    }

    function send_img(image) {
        //Make this url, in a different file, so will be accessed from anywhere
        const url = "http://localhost:8000/saveImage";
        fetch(url, {
            method: "POST",
            body: JSON.stringify(image),
            headers:{
                'Content-Type': 'application/json'
              }
            }).then(res => res.json())
            .catch(error => console.error('Error:', error))
            .then(response => console.log('Success:', response));
    }
})();