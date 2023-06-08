import * as React from "react";
import '../CSS/FileImport.css';
import { useNavigate } from "react-router-dom";
import { getConfigs, setRequirements } from "../setConfigs";
import { CircularProgress, Dialog, DialogTitle, Box } from "@mui/material";
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import axios from "axios";


const FileImport = () => {

    const navigate = useNavigate();

    const [file, setFile] = React.useState([]);
    const [isLoading, setIsLoading] = React.useState(false);
    const [done, setDone] = React.useState(false);

    let configs = [];

    const runScript = () => {
        setIsLoading(true)
        axios.post('http://localhost:8000/runScript/FILE')
            .then((response) => {
                setRequirements(response.data.text);
                setDone(true)
                setTimeout(() => {
                    handleClose()
                }, 2000)
                setTimeout(() => {
                    navigate("/configs/req-input/test-set-labeling")
                }, 1000)
            })
            .catch((e) => {
                console.log('Upload Error')
            })
    };

    const onInputChange = (e) => {
        setFile(e.target.files)
    };

    const handleClose = () => {
        setIsLoading(false)
    }

    const onSubmit = (e) => {
        e.preventDefault();

        configs.push(getConfigs());

        const data = new FormData();

        for (let i = 0; i < file.length; i++) {
            data.append('file', file[i]);
        }

        axios.post('http://localhost:8000/upload', data)
            .then((response) => {
                console.log('Upload Success');
            })
            .catch((e) => {
                console.log('Upload Error')
            })
    };

    const returnHome = () => {
        navigate("/configs");
    }

    return (
        <div className="UI">
            <div className='Header'>
                <div className="Logo">
                    <img src={require('../imgs/logotipo-removebg-preview.png')} className="logo" onClick={returnHome} />
                </div>
            </div>

            <div className="UploadFile" >
                <h1 className='Title'>Requirements Input</h1>
                <br />

                <p>Use the button below to import your file*.<br />
                    Then click "Submit File" to confirm the upload.</p>
                <form method="post" action="#" id="#" onSubmit={onSubmit} color="#0e3c45">

                    <input type="file" className="form-control" color="#0e3c45" onChange={onInputChange} />
                    <br />
                    <br />
                    <div className="Submit">
                        <button className="SubmiteButton">Submit File</button>
                    </div>
                </form>

                <p className="NOTE">*The file must be in .txt format, and the requirements separated by ';'.</p>

                <br />
                <br />

                <button onClick={runScript} className='ConfirmConfs'>
                    Confirm
                </button>

                <Dialog open={isLoading} onClose={handleClose} maxWidth="lg" >
                    <DialogTitle alignItems="center">
                        {done ? <CheckCircleIcon sx={{ fontSize: 200, paddingLeft: "40px" }} color="success" /> : (<Box sx={{ paddingLeft: "40px" }}>
                            <CircularProgress size={200} />
                        </Box>)}
                    </DialogTitle>
                    <DialogTitle>
                        {"Uploading Your Requirements..."}
                    </DialogTitle>
                </Dialog>
            </div>
        </div>
    )
}

export default FileImport;