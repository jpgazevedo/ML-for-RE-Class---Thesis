import * as React from "react";
import '../CSS/TestSetLabeling.css';
import { useNavigate } from "react-router-dom";
import { getRequirements, setRequirements } from "../setConfigs";
import { CircularProgress, Dialog, DialogTitle, Box } from "@mui/material";
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import axios from "axios"

const TestSetLabeling = () => {

    const REQTest = getRequirements();

    const [isLoading, setIsLoading] = React.useState(false);
    const [done, setDone] = React.useState(false);

    var selections = []

    REQTest.map((req) => (
        selections.push({ req: req, label: "F" })
    ))

    const classes = [
        { label: 'Functional', value: 'F' },
        { label: 'Usability', value: 'US' },
        { label: 'Security', value: 'SE' },
        { label: 'Reliability', value: 'RE' },
        { label: 'Maintainailibty', value: 'MN' },
        { label: 'Portability', value: 'PO' },
        { label: 'Performance', value: 'PE' },
        { label: 'Compatibility', value: 'CO' }
    ]

    const navigate = useNavigate();

    const next = () => {
        var labels = []

        selections.map((label) => (
            labels.push(label.label)
        ))

        setIsLoading(true)

        axios.post('http://localhost:8000/runTestLabelScript', { content: labels })
            .then((response) => {

                setRequirements(response.data.text);
                setDone(true)
                setTimeout(() => {
                  handleClose()
                }, 2000)
                setTimeout(() => {
                    navigate("/configs/req-input/test-set-labeling/train-batch-labeling");
                }, 1000) 
            })
            .catch((e) => {
                console.log('Upload Error')
            })
    }

    const returnHome = () => {
        navigate("/configs");
    }

    const handleLabelChange = (e) => {
        selections[e.target.id].label = e.target.value
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
            </div>
            <div className='AUX1'>
                <div className="ReClassify" >
                    <h1 className='TitleConfs'>Test Set Labeling</h1>
                    <p>Please label the following Requirements to be used as test set:</p>
                    {selections.map((row, index) => (
                        <div class="row" >
                            <div className="RE" >
                                <p>
                                    {row.req}
                                </p>
                            </div>
                            <select className='Label' id={index} onChangeCapture={handleLabelChange}>
                                {classes.map((option) => (
                                    <option value={option.value} >{option.label}</option>
                                ))}
                            </select>
                        </div>
                    ))}
                </div>
                <br />
                <br />
                <button className="ConfirmConfs" onClick={next}>
                    Confirm
                </button>
              
                <Dialog open={isLoading} onClose={handleClose} maxWidth="lg" >
                    <DialogTitle alignItems="center">
                        {done ? <CheckCircleIcon sx={{ fontSize: 200, paddingLeft:"40px"}} color="success" /> : (<Box sx={{paddingLeft:"40px"}}>
                            <CircularProgress size={200}   />
                        </Box>)}
                    </DialogTitle>
                    <DialogTitle>
                    {"Labeling Test Set Requirements..."}
                    </DialogTitle>
                </Dialog>
            </div>
        </div>
    )
}

export default TestSetLabeling;