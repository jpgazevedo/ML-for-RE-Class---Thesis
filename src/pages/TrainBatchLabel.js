import * as React from "react";
import '../CSS/TrainBatchLabel.css';
import { useNavigate } from "react-router-dom";
import { getRequirements, getConfigs, setConfigs, setResults } from "../setConfigs";
import { CircularProgress, Dialog, DialogTitle, Box } from "@mui/material";
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import axios from "axios"


const TrainBatchLabel = () => {
    const REQTest = getRequirements();

    var selections = []

    const [isLoading, setIsLoading] = React.useState(false);
    const [done, setDone] = React.useState(false);

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
        var initialConfigs = getConfigs()

        var labels = [initialConfigs[0], initialConfigs[2], initialConfigs[3]]

        selections.map((label) => (
            labels.push(label.label)
        ))

        setIsLoading(true)

        axios.post('http://localhost:8000/runTrainBatchScript', { content: labels })
            .then((response) => {
                setResults(response.data.text)
                setDone(true)
                setTimeout(() => {
                    handleClose()
                }, 2000)
                setTimeout(() => {
                    navigate("/configs/req-input/test-set-labeling/train-batch-labeling/stopFlg");
                }, 1000)
            })
            .catch((e) => {
                console.log('Upload Error')
            })


        setConfigs(initialConfigs[0], initialConfigs[1], initialConfigs[2], 'N')
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
                    <h1 className='TitleConfs'>Training Batch Labeling</h1>
                    <p>Please label the following Requirements to be added to the training set:</p>
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
                        {done ? <CheckCircleIcon sx={{ fontSize: 200, paddingLeft: "50px" }} color="success" /> : (<Box sx={{ paddingLeft: "50px" }}>
                            <CircularProgress size={200} />
                        </Box>)}
                    </DialogTitle>
                    <DialogTitle>
                        {"Training Machine Learning Model..."}
                    </DialogTitle>
                </Dialog>
            </div>
        </div>
    )
}

export default TrainBatchLabel;