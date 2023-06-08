import * as React from "react";
import '../CSS/ManualImport.css';
import { setRequirements } from "../setConfigs";
import { useNavigate } from "react-router-dom";
import { CircularProgress, Dialog, DialogTitle, Box } from "@mui/material";
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import axios from "axios"


const ReqImports = () => {

    const navigate = useNavigate();

    const [data, setDataSet] = React.useState('');
    const [isLoading, setIsLoading] = React.useState(false);
    const [done, setDone] = React.useState(false);

    const next = async () => {
        setIsLoading(true)
        axios.post('http://localhost:8000/runScript/MAN', { content: data })
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
    }

    const returnHome = () => {
        navigate("/configs");
    }

    const handleClose = () => {
        setIsLoading(false)
    }

    return (
        <div className="UI">
            <div className='Header'>
                <div className="Logo">
                    <img src={require('../imgs/logotipo-removebg-preview.png')} className="logo" onClick={returnHome} />
                </div>
            </div>,
            <div className='TextArea'>
                <h1 className='Title'>Requirements Input</h1>
                <label>
                    Insert your Requirements here, separated with a ';':
                    <br />
                    <textarea name="postContent" rows={30} cols={150} className='postContent' onChange={(e) => setDataSet(e.target.value)} value={data} />
                </label>
                <br />
                <br />
                <button onClick={next} className='ConfirmConfs'>
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
            <div className="AUX"></div>
        </div>
    )
}

export default ReqImports;