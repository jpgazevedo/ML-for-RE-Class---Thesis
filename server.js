const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const express = require('express');
const fs = require('fs')

const app = express();

app.use(cors());
app.use(express.json());

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, './Python Scripts/Files')
    },
    filename: (req, file, cb) => {
        cb(null, 'initialData.txt')
    }
});

const upload = multer({ storage }).array('file');

app.post('/upload', (req, res) => {
    upload(req, res, (err) => {
        if (err) {
            return res.status(500).json(err)
        }

        return res.status(200).send(req.files)
    })
});

app.post('/runTestLabelScript', (req, res) => {

    var content = req.body.content
    
    let python_process = spawn('python', ['./Python Scripts/readTestLabels.py', content]);
    

    python_process.stdout.on('data', function (data) {
        console.log('Data received from python script:', data.toString());
        return res.status(200).json({ text: data.toString() });
    });

    python_process.stderr.on('data', function (data) {
        console.log(data.toString());
    });
});


app.post('/runTrainBatchScript', (req, res) => {
    var content = req.body.content
    
    let python_process = spawn('python', ['./Python Scripts/runMLModel.py', content]);
    

    python_process.stdout.on('data', function (data) {
        console.log('Data received from python script:', data.toString());
        return res.status(200).json({ text: data.toString() });
    });

    python_process.stderr.on('data', function (data) {
        console.log(data.toString());
    });
});

app.post('/runNextTrainBatchScript', (req, res) => {
    var content = req.body.content

    let python_process = spawn('python', ['./Python Scripts/getNextTrainBatch.py', content]);

    python_process.stdout.on('data', function (data) {
        console.log('Data received from python script:', data.toString());
        return res.status(200).json({ text: data.toString() });
    });

    python_process.stderr.on('data', function (data) {
        console.log(data.toString());
    });
});

app.post('/runFinalClassification', (req, res) => {
    var content = req.body.content

    let python_process = spawn('python', ['./Python Scripts/finalClassificationNFile.py', content]);

    python_process.stdout.on('data', function (data) {
        console.log('Data received from python script:', data.toString());
        return res.status(200).json({ text: data.toString() });
    });

    python_process.stderr.on('data', function (data) {
        console.log(data.toString());
    });
});

app.post('/runScript/:ImportFlg', (req, res) => {
    const flag = req.params.ImportFlg
    var content = req.body.content
    let python_process = null

    


    if (flag === 'MAN') {
        fs.writeFile('./Python Scripts/Files/DataSet.txt', content, (err) => {
          
            // In case of a error throw err.
            if (err) throw err;
        })
        python_process = spawn('python', ['./Python Scripts/readTextAreaAndWriteFiles.py']);
    }
    else {
        python_process = spawn('python', ['./Python Scripts/readAndWriteFiles.py']);
    }

    python_process.stdout.on('data', function (data) {
        console.log('Data received from python script:', data.toString());
        return res.status(200).json({ text: data.toString() });
    });

    python_process.stderr.on('data', function (data) {
        console.log(data.toString());
    });
});


app.listen(8000, () => {
    console.log('Server is running on port 8000')
});