const express = require('express');
const path = require('path');
const fs = require('fs');
// const multer = require('multer');
// const upload = multer({dest: __dirname + '/uploads/images'});

const app = express();
const PORT = 3000;

var directoryPath = path.join(__dirname, 'images', 'personal');

fs.readdir(directoryPath, function(err, files){
    if(err) {
        return console.log('Unable to scan directory: ' + err);
    }

    files.forEach(function (file) {
        console.log(file);
    });
});


app.use(express.static('html'));
app.use(express.static('images'));
app.use(express.static('css'))

app.listen(PORT, () => {
    console.log('Listening at ' + PORT );
});
    